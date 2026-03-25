<?php

namespace Modules\Accounting\Listeners;

use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Contracts\Queue\ShouldQueue;
use App\BusinessLocation;
use App\Transaction;

class MapPaymentTransaction
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
        $payment = $event->transactionPayment;
        
        //if payment is deleted then delete the mapping
        if(isset($event->isDeleted) && $event->isDeleted){
            $accountingUtil = new \Modules\Accounting\Utils\AccountingUtil();
            $accountingUtil->deleteMap($payment->transaction_id, null);
            return;
        }

        if(empty($payment->transaction_id)){
            return;
        }

        $transaction = Transaction::find($payment->transaction_id);

        if($transaction->type == 'purchase'){
            $type = 'purchase_payment';
        } elseif($transaction->type == 'sell'){
            $type = 'sell_payment';
        } else {
            return;
        }

        //get location setting
        $business_location = BusinessLocation::find($transaction->location_id);
        $accounting_default_map = json_decode($business_location->accounting_default_map, true);

        //check if default map is set or not, if set the proceed.
        $deposit_to = isset($accounting_default_map[$type]['deposit_to']) ? $accounting_default_map[$type]['deposit_to'] : null;
        $payment_account = isset($accounting_default_map[$type]['payment_account']) ? $accounting_default_map[$type]['payment_account'] : null;

        if(!isset($event->isDeleted) || !$event->isDeleted){

            //Do the mapping
            if(!is_null($deposit_to) && !is_null($payment_account)){

                $payment_id = $payment->id;
                $user_id = request()->session()->get('user.id');
                $business_id = $transaction->business_id;
                
                $accountingUtil = new \Modules\Accounting\Utils\AccountingUtil();
                $accountingUtil->saveMap($type, $payment_id, $user_id, $business_id, $deposit_to, $payment_account);
            }
        }
    }
}
