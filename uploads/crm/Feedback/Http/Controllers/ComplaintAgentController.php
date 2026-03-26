<?php

namespace Modules\Feedback\Http\Controllers;

use App\Helper\Reply;
use App\Models\User;
use Illuminate\Http\Request;
use App\Http\Controllers\AccountBaseController;
use Modules\Feedback\Http\Requests\StoreAgentGroup;
use Modules\Feedback\Entities\FeedbackAgentGroups;
use Modules\Feedback\Entities\FeedbackGroup;

class FeedbackAgentController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'feedback::modules.feedbackAgent';
        $this->activeSettingMenu = 'ticket_settings';
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        $this->employees = User::doesntHave('agentFeedback')
            ->join('role_user', 'role_user.user_id', '=', 'users.id')
            ->join('roles', 'roles.id', '=', 'role_user.role_id')
            ->select('users.id', 'users.name', 'users.email', 'users.created_at')
            ->where('roles.name', 'employee')
            ->get();
        $this->groups = FeedbackGroup::all();
        return view('feedback::feedback-settings.create-agent-modal', $this->data);

    }

    public function store(StoreAgentGroup $request)
    {
        $users = $request->user_id;

        foreach ($users as $user) {
            $agent = new FeedbackAgentGroups();
            $agent->agent_id = $user;
            $agent->group_id = $request->group_id;
            $agent->added_by = user()->id;
            $agent->save();
        }

        if (request()->ajax()) {
            $groups = FeedbackGroup::with('enabledAgents', 'enabledAgents.user')->get();
            $agentList = '';

            foreach ($groups as $group) {
                if (count($group->enabledAgents) > 0) {

                    $agentList .= '<optgroup label="' . $group->group_name . '">';

                    foreach ($group->enabledAgents as $agent) {
                        $agentList .= '<option value="' . $agent->user->id . '">' . $agent->user->name . ' [' . $agent->user->email . ']' . '</option>';
                    }

                    $agentList .= '</optgroup>';
                }
            }

            return Reply::successWithData(__('messages.recordSaved'), ['teamData' => $agentList]);
        }

        return Reply::success(__('messages.recordSaved'));
    }

    /**
     * Update the specified resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function update(Request $request, $id)
    {
        $agent = FeedbackAgentGroups::findOrFail($id);
        $agent->status = $request->status;
        $agent->last_updated_by = user()->id;
        $agent->save();

        return Reply::success(__('messages.updateSuccess'));
    }

    public function updateGroup(Request $request, $id)
    {
        $agent = FeedbackAgentGroups::findOrFail($id);
        $agent->group_id = $request->groupId;
        $agent->last_updated_by = user()->id;
        $agent->save();

        return Reply::success(__('messages.updateSuccess'));
    }

    /**
     * Remove the specified resource from storage.
     *
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function destroy($id)
    {
        FeedbackAgentGroups::destroy($id);

        return Reply::success(__('messages.agentRemoveSuccess'));
    }

}
