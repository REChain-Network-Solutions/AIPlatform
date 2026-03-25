<?php

namespace Modules\Accounting\Entities;

use Illuminate\Database\Eloquent\Model;

class AccountingAccount extends Model
{
    /**
     * The attributes that aren't mass assignable.
     *
     * @var array
     */
    protected $guarded = ['id'];

    public function child_accounts()
    {
        return $this->hasMany(\Modules\Accounting\Entities\AccountingAccount::class, 'parent_account_id');
    }

    // public function account_type()
    // {
    //     return $this->belongsTo(\Modules\Accounting\Entities\AccountingAccountType::class, 'account_type_id');
    // }

    public function account_sub_type()
    {
        return $this->belongsTo(\Modules\Accounting\Entities\AccountingAccountType::class, 'account_sub_type_id');
    }

    public function detail_type()
    {
        return $this->belongsTo(\Modules\Accounting\Entities\AccountingAccountType::class, 'detail_type_id');
    }

    /**
     * Accounts Dropdown
     *
     * @param  int  $business_id
     * @return array
     */
    public static function forDropdown($business_id, $with_data = false, $q = '')
    {
        $query = AccountingAccount::where('accounting_accounts.business_id', $business_id)
                        ->where('status', 'active');
        if ($with_data) {
            $account_types = AccountingAccountType::accounting_primary_type();

            if (! empty($q)) {
                $query->where('accounting_accounts.name', 'like', "%{$q}%");
            }
            $accounts = $query->leftJoin('accounting_account_types as at', 'at.id', '=', 'accounting_accounts.account_sub_type_id')
                ->select('accounting_accounts.name', 'accounting_accounts.id', 'at.name as sub_type',
                 'accounting_accounts.account_primary_type', 'at.business_id as sub_type_business_id')
                ->get();

            foreach ($accounts as $k => $v) {
                $accounts[$k]->account_primary_type = ! empty($account_types[$v->account_primary_type]) ?
                $account_types[$v->account_primary_type]['label'] : $v->account_primary_type;

                $accounts[$k]->sub_type = ! empty($v->sub_type_business_id) ? $v->sub_type : __('accounting::lang.'.$v->sub_type);
            }

            return $accounts;
        } else {
            return $query->pluck('name', 'id');
        }
    }
}
