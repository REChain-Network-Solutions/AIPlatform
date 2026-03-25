<?php

namespace Modules\WorkOrders\Http\Controllers;

use Illuminate\Routing\Controller;

class WorkOrdersController extends Controller
{
    public function index()
    {
        return view('workorders::index');
    }
}
