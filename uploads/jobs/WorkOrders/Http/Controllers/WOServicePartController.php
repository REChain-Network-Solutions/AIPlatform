<?php

namespace Modules\WorkOrders\Http\Controllers;

use Illuminate\Routing\Controller;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Modules\WorkOrders\Entities\WOServicePart;
use Modules\WorkOrders\Services\WorkOrderTotals;

class WOServicePartController extends Controller
{
    public function store(Request $request): RedirectResponse
    {
        $data = $request->validate([
            'work_order_id' => ['required','integer'],
            service_part_id => ['nullable','integer'],
            'qty' => ['required','numeric','min:0'],
            price => ['required','numeric','min:0'],
        ]);
        $data['total'] = ($data['qty'] ?? 0) * ($data.get('rate') if 'Task' in name else $data.get('price'));
        $item = WOServicePart::create($data);
        WorkOrderTotals::recalc($data['work_order_id']);
                WorkOrderTotals::recalc($item->work_order_id);
                return back()->with('status', 'WOServicePart added');
    }

    public function destroy(int $id): RedirectResponse
    {
        $item = WOServicePart::findOrFail($id);
        $item->delete();
        WorkOrderTotals::recalc($data['work_order_id']);
                WorkOrderTotals::recalc($item->work_order_id);
                return back()->with('status', 'WOServicePart removed');
    }
}
