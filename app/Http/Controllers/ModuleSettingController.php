<?php

namespace App\Http\Controllers;

use App\Helper\Reply;
use App\Models\ModuleSetting;
use Illuminate\Http\Request;

class ModuleSettingController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'app.menu.moduleSettings';
        $this->activeSettingMenu = 'module_settings';
        $this->middleware(function ($request, $next) {
            abort_403(!(user()->permission('manage_module_setting') == 'all'));
            return $next($request);
        });
    }

    public function index()
    {

        $tab = request('tab');

        $this->modulesData = match ($tab) {
            'employee' => ModuleSetting::where('module_name', '<>', 'settings')->where('is_allowed', 1)->where('type', 'employee')->get(),
            'client' => ModuleSetting::where('module_name', '<>', 'settings')->where('is_allowed', 1)->where('type', 'client')->get(),
            default => ModuleSetting::where('module_name', '<>', 'settings')->where('is_allowed', 1)->where('type', 'admin')->orderBy('module_name')->get(),
        };

        $this->view = 'module-settings.ajax.modules';
        $this->activeTab = $tab ?: 'admin';

        if (request()->ajax()) {
            $html = view($this->view, $this->data)->render();
            return Reply::dataOnly(['status' => 'success', 'html' => $html, 'title' => $this->pageTitle, 'activeTab' => $this->activeTab]);
        }

        return view('module-settings.index', $this->data);
    }

    public function update(Request $request, $id)
    {
        $setting = ModuleSetting::findOrFail($id);
        $setting->status = $request->status;
        $setting->save();

        return Reply::success(__('messages.updateSuccess'));
    }

}
