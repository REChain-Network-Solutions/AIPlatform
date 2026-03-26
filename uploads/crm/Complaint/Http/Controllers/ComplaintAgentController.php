<?php

namespace Modules\Complaint\Http\Controllers;

use App\Helper\Reply;
use App\Models\User;
use Illuminate\Http\Request;
use App\Http\Controllers\AccountBaseController;
use Modules\Complaint\Http\Requests\StoreAgentGroup;
use Modules\Complaint\Entities\ComplaintAgentGroups;
use Modules\Complaint\Entities\ComplaintGroup;

class ComplaintAgentController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'complaint::modules.complaintAgent';
        $this->activeSettingMenu = 'ticket_settings';
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        $this->employees = User::doesntHave('agentComplaint')
            ->join('role_user', 'role_user.user_id', '=', 'users.id')
            ->join('roles', 'roles.id', '=', 'role_user.role_id')
            ->select('users.id', 'users.name', 'users.email', 'users.created_at')
            ->where('roles.name', 'employee')
            ->get();
        $this->groups = ComplaintGroup::all();
        return view('complaint::complaint-settings.create-agent-modal', $this->data);

    }

    public function store(StoreAgentGroup $request)
    {
        $users = $request->user_id;

        foreach ($users as $user) {
            $agent = new ComplaintAgentGroups();
            $agent->agent_id = $user;
            $agent->group_id = $request->group_id;
            $agent->added_by = user()->id;
            $agent->save();
        }

        if (request()->ajax()) {
            $groups = ComplaintGroup::with('enabledAgents', 'enabledAgents.user')->get();
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
        $agent = ComplaintAgentGroups::findOrFail($id);
        $agent->status = $request->status;
        $agent->last_updated_by = user()->id;
        $agent->save();

        return Reply::success(__('messages.updateSuccess'));
    }

    public function updateGroup(Request $request, $id)
    {
        $agent = ComplaintAgentGroups::findOrFail($id);
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
        ComplaintAgentGroups::destroy($id);

        return Reply::success(__('messages.agentRemoveSuccess'));
    }

}
