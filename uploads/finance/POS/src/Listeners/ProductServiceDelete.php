<?php

namespace Workdo\Pos\Listeners;

use App\Events\DeleteProductService;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Contracts\Queue\ShouldQueue;
use App\Models\PurchaseProduct;
use App\Models\WarehouseProduct;



class ProductServiceDelete
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
    public function handle(DeleteProductService $event)
    {
        $purchase = PurchaseProduct::where('product_id',$event->id)->get();
        if(count($purchase) == 0){
           WarehouseProduct::where('product_id',$event->id)->delete();
        }
        if(count($purchase) != 0)
        {
            return 'false';
        }
    }
}
