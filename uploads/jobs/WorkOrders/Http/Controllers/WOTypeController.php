<?php

namespace Modules\WorkOrders\Http\Controllers;

use Modules\WorkOrders\Entities\WOType;
use Illuminate\Http\Request;

class WOTypeController extends Controller
{

    public function index()
    {
        if (\Auth::user()->can('manage wo type')) {
            $woTypes = WOType::where('parent_id', parentId())->get();
            return view('wo_type.index', compact('woTypes'));
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }
    }


    public function create()
    {
        return view('wo_type.create');
    }


    public function store(Request $request)
    {
        if (\Auth::user()->can('create wo type')) {
            $validator = \Validator::make(
                $request->all(), [
                    'type' => 'required',
                ]
            );
            if ($validator->fails()) {
                $messages = $validator->getMessageBag();
                return redirect()->back()->with('error', $messages->first());
            }

            $wOType = new WOType();
            $wOType->type = $request->type;
            $wOType->parent_id =parentId();
            $wOType->save();

            return redirect()->route('wo-type.index')->with('success', __('WO Type successfully created.'));
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }
    }


    public function show(WOType $wOType)
    {
        //
    }


    public function edit($id)
    {
        $wOType=WOType::find($id);
        return view('wo_type.edit',compact('wOType'));
    }


    public function update(Request $request, $id)
    {
        if (\Auth::user()->can('edit wo type')) {
            $validator = \Validator::make(
                $request->all(), [
                    'type' => 'required',
                ]
            );
            if ($validator->fails()) {
                $messages = $validator->getMessageBag();
                return redirect()->back()->with('error', $messages->first());
            }
            $wOType=WOType::find($id);
            $wOType->type = $request->type;
            $wOType->save();

            return redirect()->route('wo-type.index')->with('success', __('WO Type successfully updated.'));
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }
    }


    public function destroy($id)
    {
        if (\Auth::user()->can('delete wo type')) {
            $wOType=WOType::find($id);
            $wOType->delete();
            return redirect()->route('wo-type.index')->with('success', __('WO Type successfully deleted.'));
        } else {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }
}
