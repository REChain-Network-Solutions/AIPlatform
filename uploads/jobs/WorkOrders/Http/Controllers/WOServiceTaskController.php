<?php

namespace Modules\WorkOrders\Http\Controllers;

use Illuminate\Routing\Controller;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Modules\WorkOrders\Entities\WOServiceTask;
use Modules\WorkOrders\Services\WorkOrderTotals;

class WOServiceTaskController extends Controller
{
    public function store(Request $request): RedirectResponse
    {
        $data = $request->validate([
            'work_order_id' => ['required','integer'],
            service_task_id => ['nullable','integer'],
            'qty' => ['required','numeric','min:0'],
            rate => ['required','numeric','min:0'],
        ]);
        $data['total'] = ($data['qty'] ?? 0) * ($data.get('rate') if 'Task' in name else $data.get('price'));
        $item = WOServiceTask::create($data);
        WorkOrderTotals::recalc($data['work_order_id']);
                WorkOrderTotals::recalc($item->work_order_id);
                return back()->with('status', 'WOServiceTask added');
    }

    public function destroy(int $id): RedirectResponse
    {
        $item = WOServiceTask::findOrFail($id);
        $item->delete();
        WorkOrderTotals::recalc($data['work_order_id']);
                WorkOrderTotals::recalc($item->work_order_id);
                return back()->with('status', 'WOServiceTask removed');
    }
}
