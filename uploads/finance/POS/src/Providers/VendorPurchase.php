<?php

namespace Workdo\Pos\Providers;

use Illuminate\Support\ServiceProvider;

class VendorPurchase extends ServiceProvider
{
    /**
     * Register the service provider.
     *
     * @return void
     */


    public function boot()
    {
        view()->composer(['account::vendor.show'], function ($view) {
            if(\Auth::check())
            {
                try {
                    $ids = \Request::segment(2);
                    if(!empty($ids))
                    {

                        try {
                            $id = \Illuminate\Support\Facades\Crypt::decrypt($ids);
                            $vendor = \Workdo\Account\Entities\Vender::where('user_id',$id)->where('created_by', creatorId())->where('workspace', getActiveWorkSpace())->first();
                            if(module_is_active('Pos', $vendor->created_by))
                            {
                                $view->getFactory()->startPush('vendor_purchase_tab', view('pos::vendor-purchase.sidebar'));
                                $view->getFactory()->startPush('vendor_purchase_div', view('pos::vendor-purchase.nav_containt_div',compact('vendor')));
                            }
                        } catch (\Throwable $th)
                        {

                        }
                    }
                } catch (\Throwable $th) {
                }
            }
        });
    }
    public function register()
    {
        //
    }

    /**
     * Get the services provided by the provider.
     *
     * @return array
     */
    public function provides()
    {
        return [];
    }
}
