<?php

namespace Modules\WorkOrders\Http\Controllers;

use Illuminate\Routing\Controller;
use Illuminate\Http\RedirectResponse;
use Illuminate\View\View;
use Modules\WorkOrders\Entities\WorkOrder;
use Modules\WorkOrders\Events\WorkOrderCreated;
use Modules\WorkOrders\Events\WorkOrderUpdated;
use Modules\WorkOrders\Events\WorkOrderCompleted;
use Modules\WorkOrders\Http\Requests\StoreWorkOrderRequest;
use Modules\WorkOrders\Http\Requests\UpdateWorkOrderRequest;

class WorkOrderController extends Controller
{
    public function index(): View
    {
        $orders = WorkOrder::latest()->paginate(20);
        return view('workorders::workorders.index', compact('orders'));
    }

    public function create(): View
    {
        return view('workorders::workorders.create');
    }

    public function store(StoreWorkOrderRequest $request): RedirectResponse
    {
        $data = $request->validated();
        $order = WorkOrder::create($data);
        event(new WorkOrderCreated($order));
                return redirect()->route('workorders.orders.show', $order->id)->with('status', 'Work order created');
    }

    public function show(int $id): View
    {
        $order = WorkOrder::with(['tasks','parts','appointments'])->findOrFail($id);
        return view('workorders::workorders.show', compact('order'));
    }

    public function edit(int $id): View
    {
        $order = WorkOrder::findOrFail($id);
        return view('workorders::workorders.edit', compact('order'));
    }

    public function update(UpdateWorkOrderRequest $request, int $id): RedirectResponse
    {
        $wasDone = false;
        
    {
        $order = WorkOrder::findOrFail($id);
        $order->update($request->validated());
        if ($order->status === 'done') { event(new WorkOrderCompleted($order)); }
        event(new WorkOrderCreated($order));
                event(new WorkOrderUpdated($order));
                return redirect()->route('workorders.orders.show', $order->id)->with('status', 'Work order updated');
    }

    public function destroy(int $id): RedirectResponse
    {
        $order = WorkOrder::findOrFail($id);
        $order->delete();
        return redirect()->route('workorders.orders.index')->with('status', 'Work order deleted');
    }
}

    public function convertToProject(int $id)
    {
        $order = \Modules\WorkOrders\Entities\WorkOrder::with(['tasks'])->findOrFail($id);
        $project = $order->convertToProject();
        if (!$project) { return back()->with('error', 'Worksuite Project/Task models missing. Update config(workorders.models.*).'); }
        return redirect()->back()->with('status', 'Converted to Project ID '.$project->id);
    }
    
