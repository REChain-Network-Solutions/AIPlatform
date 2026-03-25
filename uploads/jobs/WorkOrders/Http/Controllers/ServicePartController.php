<?php

namespace Modules\WorkOrders\Http\Controllers;

use Modules\WorkOrders\Entities\ServicePart;
use Modules\WorkOrders\Entities\ServiceTask;
use Illuminate\Http\Request;

class ServicePartController extends Controller
{

    public function index()
    {
        if (\Auth::user()->can('manage service & part')) {
            $serviceParts = ServicePart::where('parent_id', parentId())->get();
            return view('service_part.index', compact('serviceParts'));
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }
    }


    public function create()
    {
        return view('service_part.create');
    }


    public function store(Request $request)
    {
        if (\Auth::user()->can('create service & part')) {
            $validator = \Validator::make(
                $request->all(), [
                    'title' => 'required',
                    'sku' => 'required',
                    'unit' => 'required',
                    'price' => 'required',
                ]
            );
            if ($validator->fails()) {
                $messages = $validator->getMessageBag();
                return redirect()->back()->with('error', $messages->first());
            }

            $servicePart = new ServicePart();
            $servicePart->title = $request->title;
            $servicePart->sku = $request->sku;
            $servicePart->price = $request->price;
            $servicePart->unit = $request->unit;
            $servicePart->description = $request->description;
            $servicePart->type = $request->type;
            $servicePart->parent_id = parentId();
            $servicePart->save();

            if ($request->type=='service' && count($request->task) > 0 && count($request->duration) > 0) {
                $tasks = $request->task;
                $durations = $request->duration;
                $task_descriptions = $request->task_description;
                foreach ($tasks as $key => $task) {
                    if(!empty($task) && !empty($durations)){
                        $serviceTask = new ServiceTask();
                        $serviceTask->service_id = $servicePart->id;
                        $serviceTask->task = $task;
                        $serviceTask->duration = $durations[$key];
                        $serviceTask->description = $task_descriptions[$key];
                        $serviceTask->save();
                    }

                }
            }
            return redirect()->route('services-parts.index')->with('success', __('Service & Part successfully created.'));
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }
    }


    public function show($id)
    {
        $servicePart=ServicePart::find($id);
        return view('service_part.show',compact('servicePart'));
    }


    public function edit($id)
    {
        $servicePart=ServicePart::find($id);
        return view('service_part.edit',compact('servicePart'));
    }


    public function update(Request $request, $id)
    {
        if (\Auth::user()->can('edit service & part')) {
            $validator = \Validator::make(
                $request->all(), [
                    'title' => 'required',
                    'sku' => 'required',
                    'unit' => 'required',
                    'price' => 'required',
                ]
            );
            if ($validator->fails()) {
                $messages = $validator->getMessageBag();
                return redirect()->back()->with('error', $messages->first());
            }
            $servicePart=ServicePart::find($id);
            $servicePart->title = $request->title;
            $servicePart->sku = $request->sku;
            $servicePart->price = $request->price;
            $servicePart->unit = $request->unit;
            $servicePart->description = $request->description;
            $servicePart->save();

            if ($servicePart->type=='service' && count($request->task) > 0 && count($request->duration) > 0) {
                $id = $request->id;
                $tasks = $request->task;
                $durations = $request->duration;
                $task_descriptions = $request->task_description;
                foreach ($tasks as $key => $task) {

                    if (isset($id[$key]) && !empty($id[$key])) {
                        $serviceTask = ServiceTask::find($id[$key]);
                    } else {
                        $serviceTask = new ServiceTask();
                        $serviceTask->service_id = $servicePart->id;
                    }

                    $serviceTask->task = $task;
                    $serviceTask->duration = $durations[$key];
                    $serviceTask->description = $task_descriptions[$key];
                    $serviceTask->save();
                }
            }

            return redirect()->route('services-parts.index')->with('success', __('Service & Part successfully updated.'));
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }
    }


    public function destroy($id)
    {
        if (\Auth::user()->can('delete service & part')) {
            $servicePart=ServicePart::find($id);
            $servicePart->delete();
            return redirect()->route('services-parts.index')->with('success', __('Service & Part successfully deleted.'));
        } else {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }

    public function taskDestroy(Request $request)
    {

        if(!empty($request->id)){
            $task = ServiceTask::find($request->id);
            $task->delete();
        }
        return 1;
    }
}
