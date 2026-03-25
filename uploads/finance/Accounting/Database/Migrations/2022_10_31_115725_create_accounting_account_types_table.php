<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
use Modules\Accounting\Entities\AccountingAccountType;

class CreateAccountingAccountTypesTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('accounting_account_types', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('name');
            $table->integer('business_id')->nullable();
            $table->integer('created_by')->nullable();
            $table->string('account_primary_type')->nullable();
            $table->string('account_type')->nullable();
            $table->bigInteger('parent_id')->nullable();
            $table->text('description')->nullable();
            $table->boolean('show_balance')->default(1);
            $table->timestamps();
        });
        $account_sub_types = [
            [
                'id' => 1,
                'name' => 'accounts_receivable',
                'show_balance' => 0,
                'account_type' => 'sub_type',
                'account_primary_type' => 'asset',
                'parent_id' => null,
            ],
            [
                'id' => 2,
                'name' => 'current_assets',
                'show_balance' => 1,
                'account_type' => 'sub_type',
                'account_primary_type' => 'asset',
                'parent_id' => null,
            ],
            [
                'id' => 3,
                'name' => 'cash_and_cash_equivalents',
                'show_balance' => 1,
                'account_type' => 'sub_type',
                'account_primary_type' => 'asset',
                'parent_id' => null,
            ],
            [
                'id' => 4,
                'name' => 'fixed_assets',
                'show_balance' => 1,
                'account_type' => 'sub_type',
                'account_primary_type' => 'asset',
                'parent_id' => null,
            ],
            [
                'id' => 5,
                'name' => 'non_current_assets',
                'show_balance' => 1,
                'account_type' => 'sub_type',
                'account_primary_type' => 'asset',
                'parent_id' => null,
            ],
            [
                'id' => 6,
                'name' => 'accounts_payable',
                'show_balance' => 0,
                'account_type' => 'sub_type',
                'account_primary_type' => 'liability',
                'parent_id' => null,
            ],
            [
                'id' => 7,
                'name' => 'credit_card',
                'show_balance' => 1,
                'account_type' => 'sub_type',
                'account_primary_type' => 'liability',
                'parent_id' => null,
            ],
            [
                'id' => 8,
                'name' => 'current_liabilities',
                'show_balance' => 1,
                'account_type' => 'sub_type',
                'account_primary_type' => 'liability',
                'parent_id' => null,
            ],
            [
                'id' => 9,
                'name' => 'non_current_liabilities',
                'show_balance' => 1,
                'account_type' => 'sub_type',
                'account_primary_type' => 'liability',
                'parent_id' => null,
            ],
            [
                'id' => 10,
                'name' => 'owners_equity',
                'show_balance' => 1,
                'account_type' => 'sub_type',
                'account_primary_type' => 'equity',
                'parent_id' => null,
            ],
            [
                'id' => 11,
                'name' => 'income',
                'show_balance' => 0,
                'account_type' => 'sub_type',
                'account_primary_type' => 'income',
                'parent_id' => null,
            ],
            [
                'id' => 12,
                'name' => 'other_income',
                'show_balance' => 0,
                'account_type' => 'sub_type',
                'account_primary_type' => 'income',
                'parent_id' => null,
            ],
            [
                'id' => 13,
                'name' => 'cost_of_sale',
                'show_balance' => 0,
                'account_type' => 'sub_type',
                'account_primary_type' => 'expenses',
                'parent_id' => null,
            ],
            [
                'id' => 14,
                'name' => 'expenses',
                'show_balance' => 0,
                'account_type' => 'sub_type',
                'account_primary_type' => 'expenses',
                'parent_id' => null,
            ],
            [
                'id' => 15,
                'name' => 'other_expense',
                'show_balance' => 0,
                'account_type' => 'sub_type',
                'account_primary_type' => 'expenses',
                'parent_id' => null,
            ],
        ];
        AccountingAccountType::insert($account_sub_types);

        $child_account_types = [
            ['name' => 'accounts_receivable', 'parent_id' => 1, 'description' => '', 'account_type' => 'detail_type'],
            ['name' => 'allowance_for_bad_debts', 'parent_id' => 2, 'description' => 'allowance_for_bad_debts_desc', 'account_type' => 'detail_type'],
            ['name' => 'assets_available_for_sale', 'parent_id' => 2, 'description' => 'assets_available_for_sale_desc', 'account_type' => 'detail_type'],
            ['name' => 'development_costs', 'parent_id' => 2, 'description' => 'development_costs_desc', 'account_type' => 'detail_type'],
            ['name' => 'employee_cash_advances', 'parent_id' => 2, 'description' => 'employee_cash_advances_desc', 'account_type' => 'detail_type'],
            ['name' => 'inventory', 'parent_id' => 2, 'description' => 'inventory_desc', 'account_type' => 'detail_type'],
            ['name' => 'investments_-_other', 'parent_id' => 2, 'description' => 'investments_-_other_desc', 'account_type' => 'detail_type'],
            ['name' => 'loans_to_officers', 'parent_id' => 2, 'description' => 'loans_to_officers_desc', 'account_type' => 'detail_type'],
            ['name' => 'loans_to_others', 'parent_id' => 2, 'description' => 'loans_to_others_desc', 'account_type' => 'detail_type'],
            ['name' => 'loans_to_shareholders', 'parent_id' => 2, 'description' => 'loans_to_shareholders_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_current_assets', 'parent_id' => 2, 'description' => 'other_current_assets_desc', 'account_type' => 'detail_type'],
            ['name' => 'prepaid_expenses', 'parent_id' => 2, 'description' => 'prepaid_expenses_desc', 'account_type' => 'detail_type'],
            ['name' => 'retainage', 'parent_id' => 2, 'description' => 'retainage_desc', 'account_type' => 'detail_type'],
            ['name' => 'undeposited_funds', 'parent_id' => 2, 'description' => 'undeposited_funds_desc', 'account_type' => 'detail_type'],
            ['name' => 'bank', 'parent_id' => 3, 'description' => 'bank_desc', 'account_type' => 'detail_type'],
            ['name' => 'cash_and_cash_equivalents', 'parent_id' => 3, 'description' => 'cash_and_cash_equivalents_desc', 'account_type' => 'detail_type'],
            ['name' => 'cash_on_hand', 'parent_id' => 3, 'description' => 'cash_on_hand_desc', 'account_type' => 'detail_type'],
            ['name' => 'client_trust_account', 'parent_id' => 3, 'description' => 'client_trust_account_desc', 'account_type' => 'detail_type'],
            ['name' => 'money_market', 'parent_id' => 3, 'description' => 'money_market_desc', 'account_type' => 'detail_type'],
            ['name' => 'rents_held_in_trust', 'parent_id' => 3, 'description' => 'rents_held_in_trust_desc', 'account_type' => 'detail_type'],
            ['name' => 'savings', 'parent_id' => 3, 'description' => 'savings_desc', 'account_type' => 'detail_type'],
            ['name' => 'accumulated_depletion', 'parent_id' => 4, 'description' => 'accumulated_depletion_desc', 'account_type' => 'detail_type'],
            ['name' => 'accumulated_depreciation_on_property,_plant_and_equipment', 'parent_id' => 4, 'description' => 'accumulated_depreciation_on_property,_plant_and_equipment_desc', 'account_type' => 'detail_type'],
            ['name' => 'buildings', 'parent_id' => 4, 'description' => 'buildings_desc', 'account_type' => 'detail_type'],
            ['name' => 'depletable_assets', 'parent_id' => 4, 'description' => 'depletable_assets_desc', 'account_type' => 'detail_type'],
            ['name' => 'furniture_and_fixtures', 'parent_id' => 4, 'description' => 'furniture_and_fixtures_desc', 'account_type' => 'detail_type'],
            ['name' => 'land', 'parent_id' => 4, 'description' => 'land_desc', 'account_type' => 'detail_type'],
            ['name' => 'leasehold_improvements', 'parent_id' => 4, 'description' => 'leasehold_improvements_desc', 'account_type' => 'detail_type'],
            ['name' => 'machinery_and_equipment', 'parent_id' => 4, 'description' => 'machinery_and_equipment_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_fixed_assets', 'parent_id' => 4, 'description' => 'other_fixed_assets_desc', 'account_type' => 'detail_type'],
            ['name' => 'vehicles', 'parent_id' => 4, 'description' => 'vehicles_desc', 'account_type' => 'detail_type'],
            ['name' => 'accumulated_amortisation_of_non-current_assets', 'parent_id' => 5, 'description' => 'accumulated_amortisation_of_non-current_assets_desc', 'account_type' => 'detail_type'],
            ['name' => 'assets_held_for_sale', 'parent_id' => 5, 'description' => 'assets_held_for_sale_desc', 'account_type' => 'detail_type'],
            ['name' => 'deferred_tax', 'parent_id' => 5, 'description' => 'deferred_tax_desc', 'account_type' => 'detail_type'],
            ['name' => 'goodwill', 'parent_id' => 5, 'description' => 'goodwill_desc', 'account_type' => 'detail_type'],
            ['name' => 'intangible_assets', 'parent_id' => 5, 'description' => 'intangible_assets_desc', 'account_type' => 'detail_type'],
            ['name' => 'lease_buyout', 'parent_id' => 5, 'description' => 'lease_buyout_desc', 'account_type' => 'detail_type'],
            ['name' => 'licences', 'parent_id' => 5, 'description' => 'licences_desc', 'account_type' => 'detail_type'],
            ['name' => 'long-term_investments', 'parent_id' => 5, 'description' => 'long-term_investments_desc', 'account_type' => 'detail_type'],
            ['name' => 'organisational_costs', 'parent_id' => 5, 'description' => 'organisational_costs_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_non-current_assets', 'parent_id' => 5, 'description' => 'other_non-current_assets_desc', 'account_type' => 'detail_type'],
            ['name' => 'security_deposits', 'parent_id' => 5, 'description' => 'security_deposits_desc', 'account_type' => 'detail_type'],
            ['name' => 'accounts_payable_(a/p)', 'parent_id' => 6, 'description' => 'accounts_payable_(a/p)_desc', 'account_type' => 'detail_type'],
            ['name' => 'credit_card', 'parent_id' => 7, 'description' => 'credit_card_desc', 'account_type' => 'detail_type'],
            ['name' => 'accrued_liabilities', 'parent_id' => 8, 'description' => 'accrued_liabilities_desc', 'account_type' => 'detail_type'],
            ['name' => 'client_trust_accounts_-_liabilities', 'parent_id' => 8, 'description' => 'client_trust_accounts_-_liabilities_desc', 'account_type' => 'detail_type'],
            ['name' => 'current_tax_liability', 'parent_id' => 8, 'description' => 'current_tax_liability_desc', 'account_type' => 'detail_type'],
            ['name' => 'current_portion_of_obligations_under_finance_leases', 'parent_id' => 8, 'description' => 'current_portion_of_obligations_under_finance_leases_desc', 'account_type' => 'detail_type'],
            ['name' => 'dividends_payable', 'parent_id' => 8, 'description' => 'dividends_payable_desc', 'account_type' => 'detail_type'],
            ['name' => 'income_tax_payable', 'parent_id' => 8, 'description' => 'income_tax_payable_desc', 'account_type' => 'detail_type'],
            ['name' => 'insurance_payable', 'parent_id' => 8, 'description' => 'insurance_payable_desc', 'account_type' => 'detail_type'],
            ['name' => 'line_of_credit', 'parent_id' => 8, 'description' => 'line_of_credit_desc', 'account_type' => 'detail_type'],
            ['name' => 'loan_payable', 'parent_id' => 8, 'description' => 'loan_payable_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_current_liabilities', 'parent_id' => 8, 'description' => 'other_current_liabilities_desc', 'account_type' => 'detail_type'],
            ['name' => 'payroll_clearing', 'parent_id' => 8, 'description' => 'payroll_clearing_desc', 'account_type' => 'detail_type'],
            ['name' => 'payroll_liabilities', 'parent_id' => 8, 'description' => 'payroll_liabilities_desc', 'account_type' => 'detail_type'],
            ['name' => 'prepaid_expenses_payable', 'parent_id' => 8, 'description' => 'prepaid_expenses_payable_desc', 'account_type' => 'detail_type'],
            ['name' => 'rents_in_trust_-_liability', 'parent_id' => 8, 'description' => 'rents_in_trust_-_liability_desc', 'account_type' => 'detail_type'],
            ['name' => 'sales_and_service_tax_payable', 'parent_id' => 8, 'description' => 'sales_and_service_tax_payable_desc', 'account_type' => 'detail_type'],
            ['name' => 'accrued_holiday_payable', 'parent_id' => 9, 'description' => 'accrued_holiday_payable_desc', 'account_type' => 'detail_type'],
            ['name' => 'accrued_non-current_liabilities', 'parent_id' => 9, 'description' => 'accrued_non-current_liabilities_desc', 'account_type' => 'detail_type'],
            ['name' => 'liabilities_related_to_assets_held_for_sale', 'parent_id' => 9, 'description' => 'liabilities_related_to_assets_held_for_sale_desc', 'account_type' => 'detail_type'],
            ['name' => 'long-term_debt', 'parent_id' => 9, 'description' => 'long-term_debt_desc', 'account_type' => 'detail_type'],
            ['name' => 'notes_payable', 'parent_id' => 9, 'description' => 'notes_payable_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_non-current_liabilities', 'parent_id' => 9, 'description' => 'other_non-current_liabilities_desc', 'account_type' => 'detail_type'],
            ['name' => 'shareholder_notes_payable', 'parent_id' => 9, 'description' => 'shareholder_notes_payable_desc', 'account_type' => 'detail_type'],
            ['name' => 'accumulated_adjustment', 'parent_id' => 10, 'description' => 'accumulated_adjustment_desc', 'account_type' => 'detail_type'],
            ['name' => 'dividend_disbursed', 'parent_id' => 10, 'description' => 'dividend_disbursed_desc', 'account_type' => 'detail_type'],
            ['name' => 'equity_in_earnings_of_subsidiaries', 'parent_id' => 10, 'description' => 'equity_in_earnings_of_subsidiaries_desc', 'account_type' => 'detail_type'],
            ['name' => 'opening_balance_equity', 'parent_id' => 10, 'description' => 'opening_balance_equity_desc', 'account_type' => 'detail_type'],
            ['name' => 'ordinary_shares', 'parent_id' => 10, 'description' => 'ordinary_shares_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_comprehensive_income', 'parent_id' => 10, 'description' => 'other_comprehensive_income_desc', 'account_type' => 'detail_type'],
            ['name' => "owner's_equity", 'parent_id' => 10, 'description' => "owner's_equity_desc", 'account_type' => 'detail_type'],
            ['name' => 'paid-in_capital_or_surplus', 'parent_id' => 10, 'description' => 'paid-in_capital_or_surplus_desc', 'account_type' => 'detail_type'],
            ['name' => 'partner_contributions', 'parent_id' => 10, 'description' => 'partner_contributions_desc', 'account_type' => 'detail_type'],
            ['name' => 'partner_distributions', 'parent_id' => 10, 'description' => 'partner_distributions_desc', 'account_type' => 'detail_type'],
            ['name' => "partner's_equity", 'parent_id' => 10, 'description' => "partner's_equity_desc", 'account_type' => 'detail_type'],
            ['name' => 'preferred_shares', 'parent_id' => 10, 'description' => 'preferred_shares_desc', 'account_type' => 'detail_type'],
            ['name' => 'retained_earnings', 'parent_id' => 10, 'description' => 'retained_earnings_desc', 'account_type' => 'detail_type'],
            ['name' => 'share_capital', 'parent_id' => 10, 'description' => 'share_capital_desc', 'account_type' => 'detail_type'],
            ['name' => 'treasury_shares', 'parent_id' => 10, 'description' => 'treasury_shares_desc', 'account_type' => 'detail_type'],
            ['name' => 'discounts/refunds_given', 'parent_id' => 11, 'description' => 'discounts/refunds_given_desc', 'account_type' => 'detail_type'],
            ['name' => 'non-profit_income', 'parent_id' => 11, 'description' => 'non-profit_income_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_primary_income', 'parent_id' => 11, 'description' => 'other_primary_income_desc', 'account_type' => 'detail_type'],
            ['name' => 'revenue_-_general', 'parent_id' => 11, 'description' => 'revenue_-_general_desc', 'account_type' => 'detail_type'],
            ['name' => 'sales_-_retail', 'parent_id' => 11, 'description' => 'sales_-_retail_desc', 'account_type' => 'detail_type'],
            ['name' => 'sales_-_wholesale', 'parent_id' => 11, 'description' => 'sales_-_wholesale_desc', 'account_type' => 'detail_type'],
            ['name' => 'sales_of_product_income', 'parent_id' => 11, 'description' => 'sales_of_product_income_desc', 'account_type' => 'detail_type'],
            ['name' => 'service/fee_income', 'parent_id' => 11, 'description' => 'service/fee_income_desc', 'account_type' => 'detail_type'],
            ['name' => 'unapplied_cash_payment_income', 'parent_id' => 11, 'description' => 'unapplied_cash_payment_income_desc', 'account_type' => 'detail_type'],
            ['name' => 'dividend_income', 'parent_id' => 12, 'description' => 'dividend_income_desc', 'account_type' => 'detail_type'],
            ['name' => 'interest_earned', 'parent_id' => 12, 'description' => 'interest_earned_desc', 'account_type' => 'detail_type'],
            ['name' => 'loss_on_disposal_of_assets', 'parent_id' => 12, 'description' => 'loss_on_disposal_of_assets_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_investment_income', 'parent_id' => 12, 'description' => 'other_investment_income_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_miscellaneous_income', 'parent_id' => 12, 'description' => 'other_miscellaneous_income_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_operating_income', 'parent_id' => 12, 'description' => 'other_operating_income_desc', 'account_type' => 'detail_type'],
            ['name' => 'tax-exempt_interest', 'parent_id' => 12, 'description' => 'tax-exempt_interest_desc', 'account_type' => 'detail_type'],
            ['name' => 'unrealised_loss_on_securities,_net_of_tax', 'parent_id' => 12, 'description' => 'unrealised_loss_on_securities,_net_of_tax_desc', 'account_type' => 'detail_type'],
            ['name' => 'cost_of_labour_-_cos', 'parent_id' => 13, 'description' => 'cost_of_labour_-_cos_desc', 'account_type' => 'detail_type'],
            ['name' => 'equipment_rental_-_cos', 'parent_id' => 13, 'description' => 'equipment_rental_-_cos_desc', 'account_type' => 'detail_type'],
            ['name' => 'freight_and_delivery_-_cos', 'parent_id' => 13, 'description' => 'freight_and_delivery_-_cos_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_costs_of_sales_-_cos', 'parent_id' => 13, 'description' => 'other_costs_of_sales_-_cos_desc', 'account_type' => 'detail_type'],
            ['name' => 'supplies_and_materials_-_cos', 'parent_id' => 13, 'description' => 'supplies_and_materials_-_cos_desc', 'account_type' => 'detail_type'],
            ['name' => 'advertising/promotional', 'parent_id' => 14, 'description' => 'advertising/promotional_desc', 'account_type' => 'detail_type'],
            ['name' => 'amortisation_expense', 'parent_id' => 14, 'description' => 'amortisation_expense_desc', 'account_type' => 'detail_type'],
            ['name' => 'auto', 'parent_id' => 14, 'description' => 'auto_desc', 'account_type' => 'detail_type'],
            ['name' => 'bad_debts', 'parent_id' => 14, 'description' => 'bad_debts_desc', 'account_type' => 'detail_type'],
            ['name' => 'bank_charges', 'parent_id' => 14, 'description' => 'bank_charges_desc', 'account_type' => 'detail_type'],
            ['name' => 'charitable_contributions', 'parent_id' => 14, 'description' => 'charitable_contributions_desc', 'account_type' => 'detail_type'],
            ['name' => 'commissions_and_fees', 'parent_id' => 14, 'description' => 'commissions_and_fees_desc', 'account_type' => 'detail_type'],
            ['name' => 'cost_of_labour', 'parent_id' => 14, 'description' => 'cost_of_labour_desc', 'account_type' => 'detail_type'],
            ['name' => 'dues_and_subscriptions', 'parent_id' => 14, 'description' => 'dues_and_subscriptions_desc', 'account_type' => 'detail_type'],
            ['name' => 'equipment_rental', 'parent_id' => 14, 'description' => 'equipment_rental_desc', 'account_type' => 'detail_type'],
            ['name' => 'finance_costs', 'parent_id' => 14, 'description' => 'finance_costs_desc', 'account_type' => 'detail_type'],
            ['name' => 'income_tax_expense', 'parent_id' => 14, 'description' => 'income_tax_expense_desc', 'account_type' => 'detail_type'],
            ['name' => 'insurance', 'parent_id' => 14, 'description' => 'insurance_desc', 'account_type' => 'detail_type'],
            ['name' => 'interest_paid', 'parent_id' => 14, 'description' => 'interest_paid_desc', 'account_type' => 'detail_type'],
            ['name' => 'legal_and_professional_fees', 'parent_id' => 14, 'description' => 'legal_and_professional_fees_desc', 'account_type' => 'detail_type'],
            ['name' => 'loss_on_discontinued_operations,_net_of_tax', 'parent_id' => 14, 'description' => 'loss_on_discontinued_operations,_net_of_tax_desc', 'account_type' => 'detail_type'],
            ['name' => 'management_compensation', 'parent_id' => 14, 'description' => 'management_compensation_desc', 'account_type' => 'detail_type'],
            ['name' => 'meals_and_entertainment', 'parent_id' => 14, 'description' => 'meals_and_entertainment_desc', 'account_type' => 'detail_type'],
            ['name' => 'office/general_administrative_expenses', 'parent_id' => 14, 'description' => 'office/general_administrative_expenses_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_miscellaneous_service_cost', 'parent_id' => 14, 'description' => 'other_miscellaneous_service_cost_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_selling_expenses', 'parent_id' => 14, 'description' => 'other_selling_expenses_desc', 'account_type' => 'detail_type'],
            ['name' => 'payroll_expenses', 'parent_id' => 14, 'description' => 'payroll_expenses_desc', 'account_type' => 'detail_type'],
            ['name' => 'rent_or_lease_of_buildings', 'parent_id' => 14, 'description' => 'rent_or_lease_of_buildings_desc', 'account_type' => 'detail_type'],
            ['name' => 'repair_and_maintenance', 'parent_id' => 14, 'description' => 'repair_and_maintenance_desc', 'account_type' => 'detail_type'],
            ['name' => 'shipping_and_delivery_expense', 'parent_id' => 14, 'description' => 'shipping_and_delivery_expense_desc', 'account_type' => 'detail_type'],
            ['name' => 'supplies_and_materials', 'parent_id' => 14, 'description' => 'supplies_and_materials_desc', 'account_type' => 'detail_type'],
            ['name' => 'taxes_paid', 'parent_id' => 14, 'description' => 'taxes_paid_desc', 'account_type' => 'detail_type'],
            ['name' => 'travel_expenses_-_general_and_admin_expenses', 'parent_id' => 14, 'description' => 'travel_expenses_-_general_and_admin_expenses_desc', 'account_type' => 'detail_type'],
            ['name' => 'travel_expenses_-_selling_expense', 'parent_id' => 14, 'description' => 'travel_expenses_-_selling_expense_desc', 'account_type' => 'detail_type'],
            ['name' => 'unapplied_cash_bill_payment_expense', 'parent_id' => 14, 'description' => 'unapplied_cash_bill_payment_expense_desc', 'account_type' => 'detail_type'],
            ['name' => 'utilities', 'parent_id' => 14, 'description' => 'utilities_desc', 'account_type' => 'detail_type'],
            ['name' => 'amortisation', 'parent_id' => 15, 'description' => 'amortisation_desc', 'account_type' => 'detail_type'],
            ['name' => 'depreciation', 'parent_id' => 15, 'description' => 'depreciation_desc', 'account_type' => 'detail_type'],
            ['name' => 'exchange_gain_or_loss', 'parent_id' => 15, 'description' => 'exchange_gain_or_loss_desc', 'account_type' => 'detail_type'],
            ['name' => 'other_expense', 'parent_id' => 15, 'description' => 'other_expense_desc', 'account_type' => 'detail_type'],
            ['name' => 'penalties_and_settlements', 'parent_id' => 15, 'description' => 'penalties_and_settlements_desc', 'account_type' => 'detail_type'],
        ];

        AccountingAccountType::insert($child_account_types);
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('accounting_account_types');
    }
}
