<?php

use Illuminate\Support\Facades\Route;
use Modules\Complaint\Http\Controllers\ComplaintFileController;
use Modules\Complaint\Http\Controllers\ComplaintTypeController;
use Modules\Complaint\Http\Controllers\ComplaintAgentController;
use Modules\Complaint\Http\Controllers\ComplaintGroupController;
use Modules\Complaint\Http\Controllers\ComplaintReplyController;
use Modules\Complaint\Http\Controllers\ComplaintChannelController;
use Modules\Complaint\Http\Controllers\ComplaintSettingController;
use Modules\Complaint\Http\Controllers\ComplaintCustomFormController;
use Modules\Complaint\Http\Controllers\ComplaintEmailSettingController;
use Modules\Complaint\Http\Controllers\ComplaintReplyTemplatesController;
use Modules\Complaint\Http\Controllers\ComplaintController;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| contains the "web" middleware group. Now create something great!
|
*/

Route::group(['middleware' => 'auth', 'prefix' => 'account'], function () {
    Route::post('complaint/refreshCount', [ComplaintController::class, 'refreshCount'])->name('complaint.refresh_count');
    Route::post('complaint/updateOtherData/{id}', [ComplaintController::class, 'updateOtherData'])->name('complaint.update_other_data');
    Route::post('complaint/apply-quick-action', [ComplaintController::class, 'applyQuickAction'])->name('complaint.apply_quick_action');
    Route::post('complaint/change-status', [ComplaintController::class, 'changeStatus'])->name('complaint.change-status');
    Route::get('complaint/create-wr/{id}/{tn}', [ComplaintController::class, 'createWR'])->name('complaint.createWR');
    Route::get('complaint/create-wo/{id}/{tn}/{wr}', [ComplaintController::class, 'createWO'])->name('complaint.createWO');
    Route::get('complaint/get-items/{id}', [ComplaintController::class, 'getItems'])->name('complaint.get-items');
    Route::resource('complaint', ComplaintController::class);


    Route::post('complaint-form/sort-fields', [ComplaintCustomFormController::class, 'sortFields'])->name('complaint-form.sort_fields');
    Route::resource('complaint-form', ComplaintCustomFormController::class);
    Route::get('complaint-files/download/{id}', [ComplaintFileController::class, 'download'])->name('complaint-files.download');
    Route::resource('complaint-files', ComplaintFileController::class);
    Route::resource('complaint-replies', ComplaintReplyController::class);

    Route::post('complaint-agents/update-group/{id}', [ComplaintAgentController::class, 'updateGroup'])->name('complaint_agents.update_group');
    Route::resource('complaint-agents', ComplaintAgentController::class);

    Route::resource('complaint-settings', ComplaintSettingController::class);
    Route::resource('complaint-groups', ComplaintGroupController::class);
    Route::resource('complaintTypes', ComplaintTypeController::class);
    Route::resource('complaintChannels', ComplaintChannelController::class);
    Route::resource('complaint-email-settings', ComplaintEmailSettingController::class);

    Route::get('complaintTemplates/fetch-template', [ComplaintReplyTemplatesController::class, 'fetchTemplate'])->name('complaintTemplates.fetchTemplate');
    Route::resource('complaintTemplates', ComplaintReplyTemplatesController::class);
});
