<?php

namespace Workdo\Taskly\Providers;

use Illuminate\Support\Facades\Auth;
use Illuminate\Support\ServiceProvider;

class ViewComposer extends ServiceProvider
{
    /**
     * Register the service provider.
     *
     * @return void
     */


    public function boot()
    {

        view()->composer(['account::customer.show','account::vendor.show'], function ($view)
        {
            if(\Auth::check())
            {
                try {
                    $ids = \Request::segment(2);
                    if(!empty($ids))
                    {
                        try {
                            $id = \Illuminate\Support\Facades\Crypt::decrypt($ids);
                            $customer = \Workdo\Account\Entities\Customer::where('user_id',$id)->where('created_by', '=', creatorId())->where('workspace', getActiveWorkSpace())->first();
                            $vendor  = \Workdo\Account\Entities\Vender::where('user_id',$id)->where('created_by', '=', creatorId())->where('workspace', getActiveWorkSpace())->first();
                            if(module_is_active('Taskly'))
                            {
                                $view->getFactory()->startPush('customer_project_tab', view('taskly::setting.sidebar'));
                                $view->getFactory()->startPush('customer_project_div', view('taskly::setting.nav_containt_div',compact('customer')));

                                $view->getFactory()->startPush('vendor_project_tab', view('taskly::vendor.sidebar'));
                                $view->getFactory()->startPush('vendor_project_div', view('taskly::vendor.nav_containt_div',compact('vendor')));
                            }



                        } catch (\Throwable $th)
                        {
                        }
                    }
                } catch (\Throwable $th) {

                }
            }
        });

        view()->composer(['invoice.create','invoice.edit','invoice.index','invoice.grid'], function ($view)
        {
            if (Auth::check() && module_is_active('Taskly')) {
                $type = request()->query('type');
                $projectsid = request()->query('project_id');
                $view->getFactory()->startPush('account_type', view('taskly::invoice.account_type' ,compact('type','projectsid')));
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
