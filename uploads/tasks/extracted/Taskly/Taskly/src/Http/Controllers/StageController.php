<?php

namespace Workdo\Taskly\Http\Controllers;

use Illuminate\Contracts\Support\Renderable;
use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Illuminate\Support\Facades\Auth;
use Workdo\Taskly\Entities\Stage;
use Workdo\Taskly\Entities\Task;
use Workdo\Taskly\Events\TaskStageSystemSetup;


class StageController extends Controller
{
    /**
     * Display a listing of the resource.
     * @return Renderable
     */
    public function index()
    {
       if(\Auth::user()->isAbleTo('taskstage manage'))
        {
            $stages    = Stage::where('workspace_id', '=', getActiveWorkSpace())->where('created_by', '=', creatorId())->orderBy('order')->get();
            if($stages->count() < 1){
                Stage::defultadd();
            }
            return view('taskly::stages.task_stage', compact('stages'));
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
        $validator = \Validator::make($request->all(), $rules, [], $attributes);
        if($validator->fails())
        {
            $messages = $validator->getMessageBag();

            return redirect()->back()->with('error', $messages->first());
        }

        $arrStages = Stage::where('workspace_id', '=', getActiveWorkSpace())->where('created_by',creatorId())->orderBy('order')->pluck('name', 'id')->all();
        $order     = 0;
        foreach($request->stages as $key => $stage)
        {

            $obj = null;
            if($stage['id'])
            {
                $obj = Stage::find($stage['id']);
                unset($arrStages[$obj->id]);
            }
            else
            {
                $obj               = new Stage();
                $obj->workspace_id = getActiveWorkSpace();
                $obj->created_by = creatorId();
            }
            $obj->name     = $stage['name'];
            $obj->color    = $stage['color'];
            $obj->order    = $order++;
            $obj->complete = 0;
            $obj->save();
        }

        $taskExist = [];
        if($arrStages)
        {
            foreach($arrStages as $id => $name)
            {
                $count = Task::where('status', '=', $id)->count();
                if($count != 0)
                {
                    $taskExist[] = $name;
                }
                else
                {
                    Stage::find($id)->delete();
                }
            }
        }

        $lastStage = Stage::where('workspace_id', '=', getActiveWorkSpace())->where('created_by',creatorId())->orderBy('order', 'desc')->first();
        if($lastStage)
        {
            $lastStage->complete = 1;
            $lastStage->save();
        }

        event(new TaskStageSystemSetup($request));

        if(empty($taskExist))
        {
            return redirect()->back()->with('success', __('Task Stage Save Successfully.!'));
        }
        else
        {
            return redirect()->back()->with('error', __('Please remove tasks from stage: ' . implode(', ', $taskExist)));
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
