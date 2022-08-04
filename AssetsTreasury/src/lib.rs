// This file is part of Substrate.

// Copyright (C) 2017-2022 Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # AssetsTreasury Pallet
//!
//! The AssetsTreasury pallet provides a "pot" of different types of fungible assets that can be managed by stakeholders in 
//! the system and a structure for making spending proposals from this pot.
//! Current states of Assets Pallet and Treasury Pallet do not accept non-fungible assets, therefore I tried to builds some
//! workarounds, but realised that to test this pallet it would be needed to modify at least one of these original
//! pallets to accept NFTs, or do an even longer workaround just that gets messy and not very precise. 
//! So unfortunatelly I gave up on that idea for now.
//! This pallet implements a single treasure with an owner and n assets, not a set of n treasuries with n assets.
//! The treasury must always have at least one asset, the default balance, here called Currency to make it easier to 
//! compare to the original Treasury pallet. 
//!
//! - [`Config`]
//! - [`Call`]
//!
//! ## Overview
//!
//! The AssetsTreasury Pallet itself provides the pot to store different types of fungible assets for a specific account, 
//! and a means for stakeholders to propose, approve, and deny expenditures. The chain will need to provide a method (e.g.
//! inflation, fees) for collecting funds.
//!
//! By way of example, the Council could vote to fund the AssetsTreasury with a portion of the block
//! reward and use the funds to pay developers.
//!
//!
//! ### Terminology
//!
//! - **Proposal:** A suggestion to allocate funds from the pot to a beneficiary.
//! - **Beneficiary:** An account who will receive the funds from a proposal iff the proposal is
//!   approved.
//! - **Deposit:** Funds that a proposer must lock when making a proposal. The deposit will be
//!   returned or slashed if the proposal is approved or rejected respectively.
//! - **Pot:** Unspent funds accumulated by the Assets-treasury pallet and set of non-fungible assets owned by the 
//! Assets-treasury.
//!
//! ## Interface
//!
//! ### Dispatchable Functions
//!
//! General spending/proposal protocol:
//! - `propose_spend` - Make a spending proposal and stake the required deposit. Can work for spending currency or NFT.
//! - `reject_proposal` - Reject a proposal, slashing the deposit.
//! - `approve_proposal` - Accept the proposal, returning the deposit.
//! - `remove_approval` - Remove an approval, the deposit will no longer be returned.
//!
//! ## GenesisConfig
//!
//! The AssetsTreasury pallet depends on the [`GenesisConfig`].

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;
#[cfg(test)]
pub mod mock;
#[cfg(test)]
mod tests;
pub mod weights;

use codec::{Decode, Encode, MaxEncodedLen, HasCompact};
use scale_info::TypeInfo;

mod extra_mutator;
pub use extra_mutator::*;
mod functions;
mod impl_fungibles;
mod impl_stored_map;
mod types;
pub use types::*;

use sp_runtime::{
	traits::{AtLeast32BitUnsigned, Bounded, CheckedAdd, CheckedSub, AccountIdConversion, Saturating, StaticLookup, Zero},
	Permill, RuntimeDebug, ArithmeticError, TokenError,
};

use sp_std::prelude::*;

use frame_support::{
	print,
	traits::{
		Currency, ReservableCurrency,
		ExistenceRequirement::KeepAlive, 
		Get, Imbalance, OnUnbalanced,
		BalanceStatus::Reserved,
		StoredMap,
		WithdrawReasons,
	},
	dispatch::{DispatchError, DispatchResult},
	ensure,
	pallet_prelude::DispatchResultWithPostInfo,
	traits::{
		tokens::{fungibles, DepositConsequence, WithdrawConsequence},
		BalanceStatus::Reserved,
		Currency, ReservableCurrency, StoredMap,
	},
	weights::Weight,
	PalletId,
};

use frame_system::Config as SystemConfig;

pub use pallet::*;
pub use weights::WeightInfo;

/// Types related to pot's mandatory funds.
pub type BalanceOf<T, I = ()> =
	<<T as Config<I>>::Currency as Currency<<T as frame_system::Config>::AccountId>>::Balance;
pub type PositiveImbalanceOf<T, I = ()> = <<T as Config<I>>::Currency as Currency<
	<T as frame_system::Config>::AccountId,
>>::PositiveImbalance;
pub type NegativeImbalanceOf<T, I = ()> = <<T as Config<I>>::Currency as Currency<
	<T as frame_system::Config>::AccountId,
>>::NegativeImbalance;

/// A trait to allow the AssetsTreasury Pallet to spend it's funds for other purposes.
/// There is an expectation that the implementer of this trait will correctly manage
/// the mutable variables passed to it:
/// * 'assetId': The assetId of the NFT to be spent. Ignore if the asset is fungible.
/// * 'assetClass': The assetClass of the NFT to be spent. Ignore if the asset is fungible.
/// * `budget_remaining`: How much available funds that can be spent by the Assets-treasury. As funds are
///   spent, you must correctly deduct from this value. 
/// * `imbalance`: Any imbalances that you create should be subsumed in here to maximize efficiency
///   of updating the total issuance. (i.e. `deposit_creating`)
/// * `total_weight`: Track any weight that your `spend_asset` implementation uses by updating this
///   value.
/// * `missed_any`: If there were items that you want to spend on, but there were not enough funds,
///   mark this value as `true`. This will prevent the Assets-treasury from burning the excess funds.
/// This is only to be used when spending the pot's original funds, not a specific asset.
#[impl_trait_for_tuples::impl_for_tuples(30)]
pub trait SpendFunds<T: Config<I>, I: 'static = ()> {
	fn spend_funds(
		budget_remaining: &mut Balance<T, I>,
		imbalance: &mut PositiveImbalanceOf<T, I>,
		total_weight: &mut Weight,
		missed_any: &mut bool,
	);
}

/// An index of a proposal. Just a `u32`.
pub type ProposalIndex = u32;

/// A spending proposal.
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
#[derive(Encode, Decode, Clone, PartialEq, Eq, MaxEncodedLen, RuntimeDebug, TypeInfo)]
pub struct Proposal<AccountId, Balance, Option<AssetId>> {
	/// The account proposing it.
	proposer: AccountId,
	/// The account to whom the payment should be made if the proposal is accepted.
	beneficiary: AccountId,
	/// The amount held on deposit (reserved) for making this proposal.
	bond: Balance,
	/// The (total) amount that should be paid if the proposal is accepted. Even if the asset spent is an NFT
	/// there must be a balance to be spent.
	value: Balance,
	//// The assetId of the NFT the proposal is for.
	asset: AssetId,
}

#[frame_support::pallet]
pub mod pallet {
	use super::*;
	use frame_support::pallet_prelude::*;
	use frame_system::pallet_prelude::*;

	#[pallet::pallet]
	#[pallet::generate_store(pub(super) trait Store)]
	//pub struct Pallet<T, I = ()>(_);
	pub struct Pallet<T, I = ()>(PhantomData<(T, I)>);

	#[pallet::config]
	pub trait Config<I: 'static = ()>: frame_system::Config {
		/// The overarching event type.
		type Event: From<Event<Self, I>> + IsType<<Self as frame_system::Config>::Event>;

		/// The units in which we record balances.
		type Balance: Member
			+ Parameter
			+ AtLeast32BitUnsigned
			+ Default
			+ Copy
			+ MaybeSerializeDeserialize
			+ MaxEncodedLen
			+ TypeInfo;

		/// Identifier for the class of asset.
		type AssetId: Member
			+ Parameter
			+ Default
			+ Copy
			+ HasCompact
			+ MaybeSerializeDeserialize
			+ MaxEncodedLen
			+ TypeInfo;

		/// The currency mechanism.
		type Currency: ReservableCurrency<Self::AccountId>;
		/// The currency mechanism
		// type Currency: Currency<Self::AccountId> + ReservableCurrency<Self::AccountId>;

		/// The basic amount of funds that must be reserved for an asset.
		#[pallet::constant]
		type AssetDeposit: Get<DepositBalanceOf<Self, I>>;

		/// The basic amount of funds that must be reserved when adding metadata to your asset.
		#[pallet::constant]
		type MetadataDepositBase: Get<DepositBalanceOf<Self, I>>;

		/// The additional funds that must be reserved for the number of bytes you store in your
		/// metadata.
		#[pallet::constant]
		type MetadataDepositPerByte: Get<DepositBalanceOf<Self, I>>;

		/// The amount of funds that must be reserved when creating a new approval.
		#[pallet::constant]
		type ApprovalDeposit: Get<DepositBalanceOf<Self, I>>;

		/// The maximum length of a name or symbol stored on-chain.
		#[pallet::constant]
		type StringLimit: Get<u32>;

		/// A hook to allow a per-asset, per-account minimum balance to be enforced. This must be
		/// respected in all permissionless operations.
		type Freezer: FrozenBalance<Self::AssetId, Self::AccountId, Self::Balance>;

		/// Additional data to be stored with an account's asset balance.
		type Extra: Member + Parameter + Default + MaxEncodedLen;
		
		/// Origin from which approvals must come.
		type ApproveOrigin: EnsureOrigin<Self::Origin>;

		/// Origin from which rejections must come.
		type RejectOrigin: EnsureOrigin<Self::Origin>;

		/// Handler for the unbalanced decrease when slashing for a rejected proposal or bounty.
		type OnSlash: OnUnbalanced<NegativeImbalanceOf<Self, I>>;

		/// Fraction of a proposal's value that should be bonded in order to place the proposal.
		/// An accepted proposal gets these back. A rejected proposal does not.
		#[pallet::constant]
		type ProposalBond: Get<Permill>;

		/// Minimum amount of funds that should be placed in a deposit for making a proposal.
		#[pallet::constant]
		type ProposalBondMinimum: Get<BalanceOf<Self, I>>;

		/// Maximum amount of funds that should be placed in a deposit for making a proposal.
		#[pallet::constant]
		type ProposalBondMaximum: Get<Option<BalanceOf<Self, I>>>;

		/// Period between successive spends.
		#[pallet::constant]
		type SpendPeriod: Get<Self::BlockNumber>;

		/// Percentage of spare funds (if any) that are burnt per spend period.
		#[pallet::constant]
		type Burn: Get<Permill>;

		/// The Assets-treasury's pallet id, used for deriving its sovereign account ID.
		#[pallet::constant]
		type PalletId: Get<PalletId>;

		/// Handler for the unbalanced decrease when Assets-treasury funds are burned.
		type BurnDestination: OnUnbalanced<NegativeImbalanceOf<Self, I>>;

		/// Runtime hooks to external pallet using Assets-treasury to compute spend asset.
		type SpendAsset: SpendAsset<Self, I>;

		/// The maximum number of approvals that can wait in the spending queue.
		///
		/// NOTE: This parameter is also used within the Bounties Pallet extension if enabled.
		#[pallet::constant]
		type MaxApprovals: Get<u32>;

		/// The origin required for approving spends from the Assets-treasury outside of the proposal
		/// process. The `Success` value is the maximum amount that this origin is allowed to
		/// spend at a time. To make things easier this origin can only spend fungible assets.
		type SpendOrigin: EnsureOrigin<Self::Origin, Success = BalanceOf<Self, I>>;

		/// Weight information for extrinsics in this pallet.
		type WeightInfo: WeightInfo;
	}

	#[pallet::storage]
	/// Details of an asset.
	pub(super) type Asset<T: Config<I>, I: 'static = ()> = StorageMap<
		_,
		Blake2_128Concat,
		T::AssetId,
		AssetDetails<T::Balance, T::AccountId, DepositBalanceOf<T, I>>,
	>;

	#[pallet::storage]
	/// The holdings of a specific account for a specific asset.
	pub(super) type Account<T: Config<I>, I: 'static = ()> = StorageDoubleMap<
		_,
		Blake2_128Concat,
		T::AssetId,
		Blake2_128Concat,
		T::AccountId,
		AssetAccountOf<T, I>,
	>;


	/// Number of proposals that have been made.
	#[pallet::storage]
	#[pallet::getter(fn proposal_count)]
	pub(crate) type ProposalCount<T, I = ()> = StorageValue<_, ProposalIndex, ValueQuery>;

	/// Proposals that have been made.
	#[pallet::storage]
	#[pallet::getter(fn proposals)]
	pub type Proposals<T: Config<I>, I: 'static = ()> = StorageMap<
		_,
		Twox64Concat,
		ProposalIndex,
		Proposal<T::AccountId, BalanceOf<T, I>, Asset<T, I>>,
		OptionQuery,
	>;

	/// Proposal indices that have been approved but not yet awarded.
	#[pallet::storage]
	#[pallet::getter(fn approvals)]
	pub type Approvals<T: Config<I>, I: 'static = ()> =
		StorageValue<_, BoundedVec<ProposalIndex, T::MaxApprovals>, ValueQuery>;

	#[pallet::genesis_config]
	pub struct GenesisConfig;

	#[cfg(feature = "std")]
	impl Default for GenesisConfig {
		fn default() -> Self {
			Self
		}
	}

	#[cfg(feature = "std")]
	impl GenesisConfig {
		/// Direct implementation of `GenesisBuild::assimilate_storage`.
		#[deprecated(
			note = "use `<GensisConfig<T, I> as GenesisBuild<T, I>>::assimilate_storage` instead"
		)]
		pub fn assimilate_storage<T: Config<I>, I: 'static>(
			&self,
			storage: &mut sp_runtime::Storage,
		) -> Result<(), String> {
			<Self as GenesisBuild<T, I>>::assimilate_storage(self, storage)
		}
	}

	#[pallet::genesis_build]
	impl<T: Config<I>, I: 'static> GenesisBuild<T, I> for GenesisConfig {
		fn build(&self) {
			// Create Assets-treasury account
			let account_id = <Pallet<T, I>>::account_id();
			let min = T::Currency::minimum_balance();
			if T::Currency::free_balance(&account_id) < min {
				let _ = T::Currency::make_free_balance_be(&account_id, min);
			}
			// Assets are not a necessary property of a pot, but an addition, 
			//so they are built without assets
		}
	}

	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config<I>, I: 'static = ()> {
		/// New proposal.
		Proposed { proposal_index: ProposalIndex },
		// The following events regard the pot's base funds.
		/// We have ended a spend period and will now allocate funds.
		Spending { budget_remaining: BalanceOf<T, I> },
		/// Some funds have been allocated.
		Awarded { proposal_index: ProposalIndex, award: BalanceOf<T, I>, account: T::AccountId },
		/// A proposal was rejected; funds were slashed.
		Rejected { proposal_index: ProposalIndex, slashed: BalanceOf<T, I> },
		/// Some of our funds have been burnt.
		Burnt { burnt_funds: BalanceOf<T, I> },
		/// Spending has finished; this is the amount that rolls over until next spend.
		Rollover { rollover_balance: BalanceOf<T, I> },
		/// Some funds have been deposited.
		Deposit { value: BalanceOf<T, I> },
		/// A new spend proposal has been approved.
		SpendApproved {
			proposal_index: ProposalIndex,
			amount: BalanceOf<T, I>,
			beneficiary: T::AccountId,
		},
		// The following events regard the pot's assets.
		/// Some assets were transferred.
		AssetSpending { asset_id: T::AssetId, amount: T::Balance },
		/// Some assets were destroyed.
		AssetAwarded { 
			proposal_index: ProposalIndex,
			 asset_id: T::AssetId, 
			 balance: T::Balance, 
			 account:  T::AccountId 
		},
		/// Some assets were deposited.
		Deposit { asset: T::AssetId, value: T::Balance },
		/// Asset funds have been approved for use
		AssetSependApproved {
			proposal_index: ProposalIndex,
			asset_id: T::AssetId,
			amount: T::Balance,
			beneficiary: T::AccountId,
		},


	}

	/// Error for the Assets-treasury pallet.
	#[pallet::error]
	pub enum Error<T, I = ()> {
		/// Proposer's balance is too low.
		InsufficientProposersBalance,
		/// No proposal or bounty at that index.
		InvalidIndex,
		/// Too many approvals in the queue.
		TooManyApprovals,
		/// The spend origin is valid but the amount it is allowed to spend is lower than the
		/// amount to be spent.
		InsufficientPermission,
		/// Proposal has not been approved.
		ProposalNotApproved,
		/// Asset is not in pot.
		AssetNotFound,
		/// Not enough of this asset.
		InsufficientProposerAsset,
	}

	#[pallet::hooks]
	impl<T: Config<I>, I: 'static> Hooks<BlockNumberFor<T>> for Pallet<T, I> {
		/// # <weight>
		/// - Complexity: `O(A)` where `A` is the number of approvals
		/// - Db reads and writes: `Approvals`, `pot account data`
		/// - Db reads and writes per approval: `Proposals`, `proposer account data`, `beneficiary
		///   account data`
		/// - The weight is overestimated if some approvals got missed.
		/// # </weight>
		fn on_initialize(n: T::BlockNumber) -> Weight {
			// Check to see if we should spend some funds!
			if (n % T::SpendPeriod::get()).is_zero() {
				Self::spend_funds()
			} else {
				0
			}
		}
	}

	#[pallet::call]
	impl<T: Config<I>, I: 'static> Pallet<T, I> {
		/// Put forward a suggestion for spending. A deposit proportional to the value
		/// is reserved and slashed if the proposal is rejected. It is returned once the
		/// proposal is awarded.
		///
		/// # <weight>
		/// - Complexity: O(1)
		/// - DbReads: `ProposalCount`, `origin account`
		/// - DbWrites: `ProposalCount`, `Proposals`, `origin account`
		/// # </weight>
		#[pallet::weight(T::WeightInfo::propose_spend())]
		pub fn propose_spend(
			origin: OriginFor<T>,
			#[pallet::compact] value: BalanceOf<T, I>,
			beneficiary: <T::Lookup as StaticLookup>::Source,
		) -> DispatchResult {
			let proposer = ensure_signed(origin)?;
			let beneficiary = T::Lookup::lookup(beneficiary)?;

			let bond = Self::calculate_bond(value);
			T::Currency::reserve(&proposer, bond)
				.map_err(|_| Error::<T, I>::InsufficientProposersBalance)?;

			let c = Self::proposal_count();
			<ProposalCount<T, I>>::put(c + 1);
			<Proposals<T, I>>::insert(c, Proposal { proposer, value, asset, beneficiary, bond });

			Self::deposit_event(Event::Proposed { proposal_index: c });
			Ok(())
		}

		#[pallet::weight(T::WeightInfo::propose_spend())]
		pub fn propose_asset_spend(
			origin: OriginFor<T>,
			#[pallet::compact] asset: T::AssetId,
			beneficiary: <T::Lookup as StaticLookup>::Source,
			#[pallet::compact] amount: T::Balance,
			bond_value: BalanceOf<T, I>,
		) -> DispatchResult {
			let origin = ensure_signed(origin)?;
			let dest = T::Lookup::lookup(beneficiary)?;

			let bond = Self::calculate_bond(bond_value);
			T::Currency::reserve(&proposer, bond)
				.map_err(|_| Error::<T, I>::InsufficientProposersBalance)?;

			let c = Self::proposal_count();
			<ProposalCount<T, I>>::put(c + 1);
			<Proposals<T, I>>::insert(c, Proposal { proposer, asset, amount, beneficiary, bond });

			Self::deposit_event(Event::Proposed { proposal_index: c });
			Ok(())
		}

		/// Reject a proposed spend. The original deposit will be slashed.
		///
		/// May only be called from `T::RejectOrigin`.
		///
		/// # <weight>
		/// - Complexity: O(1)
		/// - DbReads: `Proposals`, `rejected proposer account`
		/// - DbWrites: `Proposals`, `rejected proposer account`
		/// # </weight>
		#[pallet::weight((T::WeightInfo::reject_proposal(), DispatchClass::Operational))]
		pub fn reject_proposal(
			origin: OriginFor<T>,
			#[pallet::compact] proposal_id: ProposalIndex,
		) -> DispatchResult {
			T::RejectOrigin::ensure_origin(origin)?;

			let proposal =
				<Proposals<T, I>>::take(&proposal_id).ok_or(Error::<T, I>::InvalidIndex)?;
			let value = proposal.bond;
			let imbalance = T::Currency::slash_reserved(&proposal.proposer, value).0;
			T::OnSlash::on_unbalanced(imbalance);

			Self::deposit_event(Event::<T, I>::Rejected {
				proposal_index: proposal_id,
				slashed: value,
			});
			Ok(())
		}

		/// Approve a proposal. At a later time, the proposal will be allocated to the beneficiary
		/// and the original deposit will be returned.
		///
		/// May only be called from `T::ApproveOrigin`.
		///
		/// # <weight>
		/// - Complexity: O(1).
		/// - DbReads: `Proposals`, `Approvals`
		/// - DbWrite: `Approvals`
		/// # </weight>
		#[pallet::weight((T::WeightInfo::approve_proposal(T::MaxApprovals::get()), DispatchClass::Operational))]
		pub fn approve_proposal(
			origin: OriginFor<T>,
			#[pallet::compact] proposal_id: ProposalIndex,
		) -> DispatchResult {
			T::ApproveOrigin::ensure_origin(origin)?;

			ensure!(<Proposals<T, I>>::contains_key(proposal_id), Error::<T, I>::InvalidIndex);
			Approvals::<T, I>::try_append(proposal_id)
				.map_err(|_| Error::<T, I>::TooManyApprovals)?;
			Ok(())
		}

		/// Propose and approve a spend of Assets-treasury funds.
		///
		/// - `origin`: Must be `SpendOrigin` with the `Success` value being at least `amount`.
		/// - `amount`: The amount to be transferred from the Assets-treasury to the `beneficiary`.
		/// - `beneficiary`: The destination account for the transfer.
		///
		/// NOTE: For record-keeping purposes, the proposer is deemed to be equivalent to the
		/// beneficiary.
		#[pallet::weight(T::WeightInfo::spend())]
		pub fn spend(
			origin: OriginFor<T>,
			#[pallet::compact] amount: BalanceOf<T, I>,
			beneficiary: <T::Lookup as StaticLookup>::Source,
		) -> DispatchResult {
			let max_amount = T::SpendOrigin::ensure_origin(origin)?;
			let beneficiary = T::Lookup::lookup(beneficiary)?;

			ensure!(amount <= max_amount, Error::<T, I>::InsufficientPermission);
			ensure!(asset in Pallet::<T, I>::pot(), Error::<T, I>::AssetNotFound);
			let proposal_index = Self::proposal_count();
			Approvals::<T, I>::try_append(proposal_index)
				.map_err(|_| Error::<T, I>::TooManyApprovals)?;
			let proposal = Proposal {
				proposer: beneficiary.clone(),
				value: amount,
				beneficiary: beneficiary.clone(),
				bond: Default::default(),
			};
			Proposals::<T, I>::insert(proposal_index, proposal);
			ProposalCount::<T, I>::put(proposal_index + 1);

			Self::deposit_event(Event::SpendApproved { proposal_index, amount, beneficiary });
			Ok(())
		}

		#[pallet::weight(T::WeightInfo::spend())]
		pub fn spend_asset(
			origin: OriginFor<T>,
			#[pallet::compact] asset: T::AssetId,
			#[pallet::compact] amount: T::Balance,
			beneficiary: <T::Lookup as StaticLookup>::Source,
		) -> DispatchResult {
			let max_amount = T::SpendOrigin::ensure_origin(origin)?;
			let beneficiary = T::Lookup::lookup(beneficiary)?;

			ensure!(amount <= max_amount, Error::<T, I>::InsufficientPermission);
			ensure!(pot_asset(asset), Error::<T, I>::AssetNotFound);
			let proposal_index = Self::proposal_count();
			Approvals::<T, I>::try_append(proposal_index)
				.map_err(|_| Error::<T, I>::TooManyApprovals)?;
			let proposal = Proposal {
				proposer: beneficiary.clone(),
				value: amount,
				asset: asset,
				beneficiary: beneficiary.clone(),
				bond: Default::default(),
			};
			Proposals::<T, I>::insert(proposal_index, proposal);
			ProposalCount::<T, I>::put(proposal_index + 1);

			Self::deposit_event(Event::SpendApproved { proposal_index, amount, asset, beneficiary });
			Ok(())
		}

		/// Force a previously approved proposal to be removed from the approval queue.
		/// The original deposit will no longer be returned.
		///
		/// May only be called from `T::RejectOrigin`.
		/// - `proposal_id`: The index of a proposal
		///
		/// # <weight>
		/// - Complexity: O(A) where `A` is the number of approvals
		/// - Db reads and writes: `Approvals`
		/// # </weight>
		///
		/// Errors:
		/// - `ProposalNotApproved`: The `proposal_id` supplied was not found in the approval queue,
		/// i.e., the proposal has not been approved. This could also mean the proposal does not
		/// exist altogether, thus there is no way it would have been approved in the first place.
		#[pallet::weight((T::WeightInfo::remove_approval(), DispatchClass::Operational))]
		pub fn remove_approval(
			origin: OriginFor<T>,
			#[pallet::compact] proposal_id: ProposalIndex,
		) -> DispatchResult {
			T::RejectOrigin::ensure_origin(origin)?;

			Approvals::<T, I>::try_mutate(|v| -> DispatchResult {
				if let Some(index) = v.iter().position(|x| x == &proposal_id) {
					v.remove(index);
					Ok(())
				} else {
					Err(Error::<T, I>::ProposalNotApproved.into())
				}
			})?;

			Ok(())
		}
	}
}

impl<T: Config<I>, I: 'static> Pallet<T, I> {
	// Add public immutables and private mutables.

	/// The account ID of the Assets-treasury pot.
	///
	/// This actually does computation. If you need to keep using it, then make sure you cache the
	/// value and only call this once.
	pub fn account_id() -> T::AccountId {
		T::PalletId::get().into_account_truncating()
	}

	/// The needed bond for a proposal whose spend is `value`.
	fn calculate_bond(value: BalanceOf<T, I>) -> BalanceOf<T, I> {
		let mut r = T::ProposalBondMinimum::get().max(T::ProposalBond::get() * value);
		if let Some(m) = T::ProposalBondMaximum::get() {
			r = r.min(m);
		}
		r
	}

	/// Spend some money! returns number of approvals before spend.
	pub fn spend_funds() -> Weight {
		let mut total_weight: Weight = Zero::zero();

		let mut budget_remaining = Self::pot();
		Self::deposit_event(Event::Spending { budget_remaining });
		let account_id = Self::account_id();

		let mut missed_any = false;
		let mut imbalance = <PositiveImbalanceOf<T, I>>::zero();
		let proposals_len = Approvals::<T, I>::mutate(|v| {
			let proposals_approvals_len = v.len() as u32;
			v.retain(|&index| {
				// Should always be true, but shouldn't panic if false or we're screwed.
				if let Some(p) = Self::proposals(index) {
					if p.value <= budget_remaining {
						budget_remaining -= p.value;
						<Proposals<T, I>>::remove(index);

						// return their deposit.
						let err_amount = T::Currency::unreserve(&p.proposer, p.bond);
						debug_assert!(err_amount.is_zero());

						// provide the allocation.
						imbalance.subsume(T::Currency::deposit_creating(&p.beneficiary, p.value));

						Self::deposit_event(Event::Awarded {
							proposal_index: index,
							award: p.value,
							account: p.beneficiary,
						});
						false
					} else {
						missed_any = true;
						true
					}
				} else {
					false
				}
			});
			proposals_approvals_len
		});

		total_weight += T::WeightInfo::on_initialize_proposals(proposals_len);

		// Call Runtime hooks to external pallet using Assets-treasury to compute spend funds.
		T::SpendFunds::spend_funds(
			&mut asset,
			&mut budget_remaining,
			&mut imbalance,
			&mut total_weight,
			&mut missed_any,
		);

		if !missed_any {
			// burn some proportion of the remaining budget if we run a surplus.
			let burn = (T::Burn::get() * budget_remaining).min(budget_remaining);
			budget_remaining -= burn;

			let (debit, credit) = T::Currency::pair(burn);
			imbalance.subsume(debit);
			T::BurnDestination::on_unbalanced(credit);
			Self::deposit_event(Event::Burnt { burnt_funds: burn })
		}

		// Must never be an error, but better to be safe.
		// proof: budget_remaining is account free balance minus ED;
		// Thus we can't spend more than account free balance minus ED;
		// Thus account is kept alive; qed;
		if let Err(problem) =
			T::Currency::settle(&account_id, imbalance, WithdrawReasons::TRANSFER, KeepAlive)
		{
			print("Inconsistent state - couldn't settle imbalance for funds spent by Assets-treasury");
			// Nothing else to do here.
			drop(problem);
		}

		Self::deposit_event(Event::Rollover { rollover_balance: budget_remaining });

		total_weight
	}

	/// Return the amount of money in the pot.
	// The existential deposit is not part of the pot so Assets-treasury account never gets deleted.
	pub fn pot() -> BalanceOf<T, I>{
		T::Currency::free_balance(&Self::account_id())
			// Must never be less than 0 but better be safe.
			.saturating_sub(T::Currency::minimum_balance())
	}

	/// Return if an asset is in the pot
	// The existential deposit is not part of the pot so Assets-treasury account never gets deleted.
	pub fn pot(asset: AssetId) -> bool{
		T::Account<&Self::account_id()>::AssetAccountOf(asset)
	}
}

impl<T: Config<I>, I: 'static> OnUnbalanced<NegativeImbalanceOf<T, I>> for Pallet<T, I> {
	fn on_nonzero_unbalanced(amount: NegativeImbalanceOf<T, I>) {
		let numeric_amount = amount.peek();

		// Must resolve into existing but better to be safe.
		let _ = T::Currency::resolve_creating(&Self::account_id(), amount);

		Self::deposit_event(Event::Deposit { value: numeric_amount });
	}
}