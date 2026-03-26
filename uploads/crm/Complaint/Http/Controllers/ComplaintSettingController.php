<?php

namespace Modules\Complaint\Http\Controllers;

use App\Helper\Reply;
use App\Models\User;
use App\Http\Controllers\AccountBaseController;
use Modules\Complaint\Entities\ComplaintAgentGroups;
use Modules\Complaint\Entities\ComplaintChannel;
use Modules\Complaint\Entities\ComplaintEmailSetting;
use Modules\Complaint\Entities\ComplaintGroup;
use Modules\Complaint\Entities\ComplaintReplyTemplate;
use Modules\Complaint\Entities\ComplaintType;

class ComplaintSettingController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'app.menu.complaintSettings';
        $this->activeSettingMenu = 'complaint_settings';
        $this->middleware(function ($request, $next) {
            abort_403(!(user()->permission('add_complaint') == 'all'));
            return $next($request);
        });
    }

    /**
     * Display a listing of the resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function index()
    {
        $this->agents = ComplaintAgentGroups::with('user')->get();
        $this->employees = User::doesntHave('agent')
            ->join('role_user', 'role_user.user_id', '=', 'users.id')
            ->join('roles', 'roles.id', '=', 'role_user.role_id')
            ->select('users.id', 'users.name', 'users.email', 'users.created_at')
            ->where('roles.name', 'employee')
            ->get();

        $this->groups = ComplaintGroup::all();
        $this->ticketTypes = ComplaintType::all();
        $this->templates = ComplaintReplyTemplate::all();
        $this->channels = ComplaintChannel::all();
        $this->ticketEmailSetting = ComplaintEmailSetting::first();


        $this->view = 'complaint::complaint-settings.ajax.agent';

        $tab = request('tab');

        switch ($tab) {
        case 'type':
            $this->pageTitle = 'complaint::modules.complaintTypes';
            $this->view = 'complaint::complaint-settings.ajax.type';
            break;
        case 'channel':
            $this->pageTitle = 'complaint::modules.complaintChannels';
            $this->view = 'complaint::complaint-settings.ajax.channel';
            break;
        case 'reply-template':
            $this->pageTitle = 'complaint::modules.complaintTemplates';
            $this->view = 'complaint::complaint-settings.ajax.reply-template';
            break;
        case 'email-sync':
            $this->pageTitle = 'app.menu.emailSync';
            $this->view = 'complaint::complaint-settings.ajax.email-sync';
            break;
        default:
            $this->pageTitle = 'complaint::modules.complaintAgents';
            $this->view = 'complaint::complaint-settings.ajax.agent';
            break;
        }

        $this->activeTab = $tab ?: 'agent';

        if (request()->ajax()) {
            $html = view($this->view, $this->data)->render();
            return Reply::dataOnly(['status' => 'success', 'html' => $html, 'title' => $this->pageTitle, 'activeTab' => $this->activeTab]);
        }

        return view('complaint::complaint-settings.index', $this->data);

    }

}
