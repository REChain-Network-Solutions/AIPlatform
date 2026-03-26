<?php

namespace Modules\Feedback\Http\Controllers;

use App\Helper\Reply;
use App\Models\User;
use App\Http\Controllers\AccountBaseController;
use Modules\Feedback\Entities\FeedbackAgentGroups;
use Modules\Feedback\Entities\FeedbackChannel;
use Modules\Feedback\Entities\FeedbackEmailSetting;
use Modules\Feedback\Entities\FeedbackGroup;
use Modules\Feedback\Entities\FeedbackReplyTemplate;
use Modules\Feedback\Entities\FeedbackType;

class FeedbackSettingController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'app.menu.feedbackSettings';
        $this->activeSettingMenu = 'feedback_settings';
        $this->middleware(function ($request, $next) {
            abort_403(!(user()->permission('add_feedback') == 'all'));
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
        $this->agents = FeedbackAgentGroups::with('user')->get();
        $this->employees = User::doesntHave('agent')
            ->join('role_user', 'role_user.user_id', '=', 'users.id')
            ->join('roles', 'roles.id', '=', 'role_user.role_id')
            ->select('users.id', 'users.name', 'users.email', 'users.created_at')
            ->where('roles.name', 'employee')
            ->get();

        $this->groups = FeedbackGroup::all();
        $this->ticketTypes = FeedbackType::all();
        $this->templates = FeedbackReplyTemplate::all();
        $this->channels = FeedbackChannel::all();
        $this->ticketEmailSetting = FeedbackEmailSetting::first();


        $this->view = 'feedback::feedback-settings.ajax.agent';

        $tab = request('tab');

        switch ($tab) {
        case 'type':
            $this->pageTitle = 'feedback::modules.feedbackTypes';
            $this->view = 'feedback::feedback-settings.ajax.type';
            break;
        case 'channel':
            $this->pageTitle = 'feedback::modules.feedbackChannels';
            $this->view = 'feedback::feedback-settings.ajax.channel';
            break;
        case 'reply-template':
            $this->pageTitle = 'feedback::modules.feedbackTemplates';
            $this->view = 'feedback::feedback-settings.ajax.reply-template';
            break;
        case 'email-sync':
            $this->pageTitle = 'app.menu.emailSync';
            $this->view = 'feedback::feedback-settings.ajax.email-sync';
            break;
        default:
            $this->pageTitle = 'feedback::modules.feedbackAgents';
            $this->view = 'feedback::feedback-settings.ajax.agent';
            break;
        }

        $this->activeTab = $tab ?: 'agent';

        if (request()->ajax()) {
            $html = view($this->view, $this->data)->render();
            return Reply::dataOnly(['status' => 'success', 'html' => $html, 'title' => $this->pageTitle, 'activeTab' => $this->activeTab]);
        }

        return view('feedback::feedback-settings.index', $this->data);

    }

}
