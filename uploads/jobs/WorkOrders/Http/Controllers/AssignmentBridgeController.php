<?php

namespace Modules\WorkOrders\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Routing\Controller;

class AssignmentBridgeController extends Controller
{
    public function index($id)
    {
        // Pull assignments from Contractors module if present
        $rows = collect();
        if (class_exists(\Modules\WorksuiteContractors\Entities\WorkOrderContractorAssignment::class)) {
            $rows = \Modules\WorksuiteContractors\Entities\WorkOrderContractorAssignment::with('contractor')
                ->where('work_order_id', $id)->orderByDesc('scheduled_at')->get();
        }
        return view('workorders::widgets.assignments', ['work_order_id'=>$id, 'rows'=>$rows]);
    }

    public function goAssign($id)
    {
        // Redirect to Contractors assignment UI with pre-filled work_order_id
        return redirect()->to('/contractors/assignments?work_order_id='.(int)$id);
    }
}
