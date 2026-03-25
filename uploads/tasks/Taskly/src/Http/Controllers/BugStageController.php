<?php

namespace Workdo\Taskly\Http\Controllers;

use Illuminate\Contracts\Support\Renderable;
use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Illuminate\Support\Facades\Auth;
use Workdo\Taskly\Entities\BugReport;
use Workdo\Taskly\Entities\BugStage;
use Illuminate\Support\Facades\Validator;
use Workdo\Taskly\Events\BugStageSystemSetup;


class BugStageController extends Controller
{
    /**
     * Display a listing of the resource.
     * @return Renderable
     */
    public function index()
    {
        if(Auth::user()->isAbleTo('bugstage manage'))
        {
            $bugStages = BugStage::where('workspace_id', '=', getActiveWorkSpace())->where('created_by',creatorId())->orderBy('order')->get();
                if($bugStages->count() < 1){
                    BugStage::defultadd();
                }
            return view('taskly::stages.bug_stage', compact('bugStages'));
        }
        else
        {
            return redirect()->back()->with('error', 'permission Denied');
        }
    }
    /**
     * Show the form for creating a new resource.
     * @return Renderable
     */
    public function create()
    {

        return view('taskly::create');
    }

    /**
     * Store a newly created resource in storage.
     * @param Request $request
     * @return Renderable
     */
    public function store(Request $request)
    {
        $rules      = [
            'stages' => 'required|present|array',
        ];
        $attributes = [];
        if($request->stages)
        {

            foreach($request->stages as $key => $val)
            {
                $rules['stages.' . $key . '.name']      = 'required|max:255';
                $attributes['stages.' . $key . '.name'] = __('Stage Name');
            }
        }
        $validator = Validator::make($request->all(), $rules, [], $attributes);
        if($validator->fails())
        {
            $messages = $validator->getMessageBag();

            return redirect()->back()->with('error', $messages->first());
        }

        $arrStages = BugStage::where('workspace_id', getActiveWorkSpace())->where('created_by',creatorId())->orderBy('order')->pluck('name', 'id')->all();
        $order     = 0;
        foreach($request->stages as $key => $stage)
        {
            $obj = null;
            if($stage['id'])
            {
                $obj = BugStage::find($stage['id']);
                unset($arrStages[$obj->id]);
            }
            else
            {
                $obj               = new BugStage();
                $obj->workspace_id = getActiveWorkSpace(); 
            }
            $obj->name     = $stage['name'];
            $obj->color    = $stage['color'];
            $obj->order    = $order++;
            $obj->complete = 0;
            $obj->created_by = creatorId();
            $obj->save();
        }

        $taskExist = [];
        if($arrStages)
        {
            foreach($arrStages as $id => $name)
            {
                $count = BugReport::where('status', '=', $id)->count();
                if($count != 0)
                {
                    $taskExist[] = $name;
                }
                else
                {
                    BugStage::find($id)->delete();
                }
            }
        }

        $lastStage = BugStage::where('workspace_id', getActiveWorkSpace())->where('created_by',creatorId())->orderBy('order', 'desc')->first();
        if($lastStage)
        {
            $lastStage->complete = 1;
            $lastStage->save();
        }

        event(new BugStageSystemSetup($request));

        if(empty($taskExist))
        {
            return redirect()->back()->with('success', __('The bug stage has been created successfully.'));
        }
        else
        {
            return redirect()->back()->with('error', __('Please remove bugs from stage: ' . implode(', ', $taskExist)));
        }
    }

    /**
     * Show the specified resource.
     * @param int $id
     * @return Renderable
     */
    public function show($id)
    {
        return view('taskly::show');
    }

    /**
     * Show the form for editing the specified resource.
     * @param int $id
     * @return Renderable
     */
    public function edit($id)
    {
        return view('taskly::edit');
    }

    /**
     * Update the specified resource in storage.
     * @param Request $request
     * @param int $id
     * @return Renderable
     */
    public function update(Request $request, $id)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     * @param int $id
     * @return Renderable
     */
    public function destroy($id)
    {
        //
    }
}
