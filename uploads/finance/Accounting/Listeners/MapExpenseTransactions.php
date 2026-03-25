<?php

namespace Modules\Accounting\Listeners;

use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Contracts\Queue\ShouldQueue;
use App\BusinessLocation;

class MapExpenseTransactions
{
    /**
     * Create the event listener.
     *
     * @return void
     */
    public function __construct()
    {
        //
    }

    /**
     * Handle the event.
     *
     * @param  object  $event
     * @return void
     */
    public function handle($event)
    {
        $business_location = BusinessLocation::find($event->expense->location_id);
        $accounting_default_map = json_decode($business_location->accounting_default_map, true);
        // if expense category is selected
        if(!empty($event->expense->expense_category_id)){
            
                $deposit_to = isset($accounting_default_map['expense_'.$event->expense->expense_category_id]['deposit_to']) ? $accounting_default_map['expense_'.$event->expense->expense_category_id]['deposit_to'] : null;

                $payment_account = isset($accounting_default_map['expense_'.$event->expense->expense_category_id]['payment_account']) ? $accounting_default_map['expense_'.$event->expense->expense_category_id]['payment_account'] : null;
            // if expense category is selected but value is null 
            if(is_null($deposit_to) || is_null($payment_account)){
                $deposit_to = isset($accounting_default_map['expense']['deposit_to']) ? $accounting_default_map['expense']['deposit_to'] : null;
                $payment_account = isset($accounting_default_map['expense']['payment_account']) ? $accounting_default_map['expense']['payment_account'] : null;
            }
               
        }else{

            $deposit_to = isset($accounting_default_map['expense']['deposit_to']) ? $accounting_default_map['expense']['deposit_to'] : null;
            $payment_account = isset($accounting_default_map['expense']['payment_account']) ? $accounting_default_map['expense']['payment_account'] : null;
        }

        //if expense is deleted then delete the mapping
        if(isset($event->isDeleted) && $event->isDeleted){
            $accountingUtil = new \Modules\Accounting\Utils\AccountingUtil();
            $accountingUtil->deleteMap($event->expense->id, null);
        } else {
            if(!is_null($deposit_to) && !is_null($payment_account)){
                $type = 'expense';
                $id = $event->expense->id;
                $user_id = request()->session()->get('user.id');
                $business_id = $event->expense->business_id;
                $accountingUtil = new \Modules\Accounting\Utils\AccountingUtil();
                $accountingUtil->saveMap($type, $id, $user_id, $business_id, $deposit_to, $payment_account);
            }
        }
    }
}
