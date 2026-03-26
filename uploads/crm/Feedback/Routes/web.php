<?php

use Illuminate\Support\Facades\Route;
use Modules\Feedback\Http\Controllers\FeedbackFileController;
use Modules\Feedback\Http\Controllers\FeedbackTypeController;
use Modules\Feedback\Http\Controllers\FeedbackAgentController;
use Modules\Feedback\Http\Controllers\FeedbackGroupController;
use Modules\Feedback\Http\Controllers\FeedbackReplyController;
use Modules\Feedback\Http\Controllers\FeedbackChannelController;
use Modules\Feedback\Http\Controllers\FeedbackSettingController;
use Modules\Feedback\Http\Controllers\FeedbackCustomFormController;
use Modules\Feedback\Http\Controllers\FeedbackEmailSettingController;
use Modules\Feedback\Http\Controllers\FeedbackReplyTemplatesController;
use Modules\Feedback\Http\Controllers\FeedbackController;

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
    Route::post('feedback/refreshCount', [FeedbackController::class, 'refreshCount'])->name('feedback.refresh_count');
    Route::post('feedback/updateOtherData/{id}', [FeedbackController::class, 'updateOtherData'])->name('feedback.update_other_data');
    Route::post('feedback/apply-quick-action', [FeedbackController::class, 'applyQuickAction'])->name('feedback.apply_quick_action');
    Route::post('feedback/change-status', [FeedbackController::class, 'changeStatus'])->name('feedback.change-status');
    Route::get('feedback/create-wr/{id}/{tn}', [FeedbackController::class, 'createWR'])->name('feedback.createWR');
    Route::get('feedback/create-wo/{id}/{tn}/{wr}', [FeedbackController::class, 'createWO'])->name('feedback.createWO');
    Route::get('feedback/get-items/{id}', [FeedbackController::class, 'getItems'])->name('feedback.get-items');
    Route::resource('feedback', FeedbackController::class);


    Route::post('feedback-form/sort-fields', [FeedbackCustomFormController::class, 'sortFields'])->name('feedback-form.sort_fields');
    Route::resource('feedback-form', FeedbackCustomFormController::class);
    Route::get('feedback-files/download/{id}', [FeedbackFileController::class, 'download'])->name('feedback-files.download');
    Route::resource('feedback-files', FeedbackFileController::class);
    Route::resource('feedback-replies', FeedbackReplyController::class);

    Route::post('feedback-agents/update-group/{id}', [FeedbackAgentController::class, 'updateGroup'])->name('feedback_agents.update_group');
    Route::resource('feedback-agents', FeedbackAgentController::class);

    Route::resource('feedback-settings', FeedbackSettingController::class);
    Route::resource('feedback-groups', FeedbackGroupController::class);
    Route::resource('feedbackTypes', FeedbackTypeController::class);
    Route::resource('feedbackChannels', FeedbackChannelController::class);
    Route::resource('feedback-email-settings', FeedbackEmailSettingController::class);

    Route::get('feedbackTemplates/fetch-template', [FeedbackReplyTemplatesController::class, 'fetchTemplate'])->name('feedbackTemplates.fetchTemplate');
    Route::resource('feedbackTemplates', FeedbackReplyTemplatesController::class);
});

Route::middleware(['web','auth'])->prefix('feedback')->name('feedback.')->group(function () {
    Route::get('/nps', [\Modules\Feedback\Http\Controllers\NpsController::class, 'index'])->name('nps.index');
    Route::post('/nps', [\Modules\Feedback\Http\Controllers\NpsController::class, 'store'])->name('nps.store');
});

Route::middleware(['web','auth'])->get('/feedback/insights', [\Modules\Feedback\Http\Controllers\InsightsController::class, 'index'])->name('feedback.insights');
