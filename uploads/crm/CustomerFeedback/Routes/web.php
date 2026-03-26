<?php

use Illuminate\Support\Facades\Route;
use Modules\CustomerFeedback\Http\Controllers\AnalyticsController;
use Modules\CustomerFeedback\Http\Controllers\CsatSurveyController;
use Modules\CustomerFeedback\Http\Controllers\FeedbackAgentGroupController;
use Modules\CustomerFeedback\Http\Controllers\FeedbackChannelController;
use Modules\CustomerFeedback\Http\Controllers\FeedbackCustomFormController;
use Modules\CustomerFeedback\Http\Controllers\FeedbackEmailSettingController;
use Modules\CustomerFeedback\Http\Controllers\FeedbackFileController;
use Modules\CustomerFeedback\Http\Controllers\FeedbackGroupController;
use Modules\CustomerFeedback\Http\Controllers\FeedbackInsightsController;
use Modules\CustomerFeedback\Http\Controllers\FeedbackReplyController;
use Modules\CustomerFeedback\Http\Controllers\FeedbackReplyTemplateController;
use Modules\CustomerFeedback\Http\Controllers\FeedbackTicketController;
use Modules\CustomerFeedback\Http\Controllers\FeedbackTypeController;
use Modules\CustomerFeedback\Http\Controllers\NpsSurveyController;

Route::middleware(['web', 'auth'])->prefix('customer-feedback')->as('feedback.')->group(function () {
    Route::resource('tickets', FeedbackTicketController::class);
    Route::post('tickets/bulk', [FeedbackTicketController::class, 'bulk'])->name('tickets.bulk');
    Route::post('tickets/export', [FeedbackTicketController::class, 'export'])->name('tickets.export');
    Route::post('tickets/{ticket}/replies', [FeedbackReplyController::class, 'store'])->name('replies.store');
    Route::get('tickets/{ticket}/replies', [FeedbackReplyController::class, 'index'])->name('replies.index');
    Route::delete('replies/{reply}', [FeedbackReplyController::class, 'destroy'])->name('replies.destroy');
    Route::post('tickets/{ticket}/replies/resolve', [FeedbackReplyController::class, 'resolve'])->name('replies.resolve');
    Route::get('reply-templates/{template}', [FeedbackReplyController::class, 'template'])->name('reply-template.show');
    Route::get('files/{file}', [FeedbackFileController::class, 'show'])->name('files.show');
    Route::get('files/{file}/download', [FeedbackFileController::class, 'download'])->name('files.download');
    Route::delete('files/{file}', [FeedbackFileController::class, 'destroy'])->name('files.destroy');
    Route::prefix('surveys/nps')->as('nps.')->group(function () { Route::get('/', [NpsSurveyController::class, 'index'])->name('index'); Route::get('create', [NpsSurveyController::class, 'create'])->name('create'); Route::post('/', [NpsSurveyController::class, 'store'])->name('store'); Route::get('{survey}', [NpsSurveyController::class, 'show'])->name('show'); Route::delete('{survey}', [NpsSurveyController::class, 'destroy'])->name('destroy'); Route::post('{survey}/submit', [NpsSurveyController::class, 'submitResponse'])->withoutMiddleware('auth')->name('submit'); });
    Route::prefix('surveys/csat')->as('csat.')->group(function () { Route::get('/', [CsatSurveyController::class, 'index'])->name('index'); Route::get('create', [CsatSurveyController::class, 'create'])->name('create'); Route::post('/', [CsatSurveyController::class, 'store'])->name('store'); Route::get('{survey}', [CsatSurveyController::class, 'show'])->name('show'); Route::delete('{survey}', [CsatSurveyController::class, 'destroy'])->name('destroy'); Route::post('{survey}/submit', [CsatSurveyController::class, 'submitResponse'])->withoutMiddleware('auth')->name('submit'); });
    Route::prefix('analytics')->as('analytics.')->group(function () { Route::get('dashboard', [AnalyticsController::class, 'dashboard'])->name('dashboard'); Route::get('nps', [AnalyticsController::class, 'nps'])->name('nps'); Route::get('csat', [AnalyticsController::class, 'csat'])->name('csat'); });
    Route::prefix('insights')->as('insights.')->group(function () { Route::get('dashboard', [FeedbackInsightsController::class, 'dashboard'])->name('dashboard'); Route::get('tickets/{ticket}', [FeedbackInsightsController::class, 'getTicketInsights'])->name('ticket'); Route::post('tickets/{ticket}/analyze', [FeedbackInsightsController::class, 'analyzeTicket'])->name('analyze'); Route::get('tickets/{ticket}/sentiment', [FeedbackInsightsController::class, 'getSentiment'])->name('sentiment'); Route::get('tickets/{ticket}/category', [FeedbackInsightsController::class, 'getSuggestedCategory'])->name('category'); Route::get('tickets/{ticket}/priority', [FeedbackInsightsController::class, 'getSuggestedPriority'])->name('priority'); Route::get('tickets/{ticket}/response', [FeedbackInsightsController::class, 'getSuggestedResponse'])->name('response'); });
    Route::prefix('settings')->as('settings.')->group(function () {
        Route::get('channels', [FeedbackChannelController::class, 'index'])->name('channels.index'); Route::post('channels', [FeedbackChannelController::class, 'store'])->name('channels.store'); Route::delete('channels/{item}', [FeedbackChannelController::class, 'destroy'])->name('channels.destroy');
        Route::get('types', [FeedbackTypeController::class, 'index'])->name('types.index'); Route::post('types', [FeedbackTypeController::class, 'store'])->name('types.store'); Route::delete('types/{item}', [FeedbackTypeController::class, 'destroy'])->name('types.destroy');
        Route::get('groups', [FeedbackGroupController::class, 'index'])->name('groups.index'); Route::post('groups', [FeedbackGroupController::class, 'store'])->name('groups.store'); Route::delete('groups/{item}', [FeedbackGroupController::class, 'destroy'])->name('groups.destroy');
        Route::get('templates', [FeedbackReplyTemplateController::class, 'index'])->name('templates.index'); Route::post('templates', [FeedbackReplyTemplateController::class, 'store'])->name('templates.store'); Route::delete('templates/{item}', [FeedbackReplyTemplateController::class, 'destroy'])->name('templates.destroy');
        Route::get('forms', [FeedbackCustomFormController::class, 'index'])->name('forms.index'); Route::post('forms', [FeedbackCustomFormController::class, 'store'])->name('forms.store'); Route::delete('forms/{item}', [FeedbackCustomFormController::class, 'destroy'])->name('forms.destroy');
        Route::get('email', [FeedbackEmailSettingController::class, 'index'])->name('email.index'); Route::post('email', [FeedbackEmailSettingController::class, 'store'])->name('email.store');
        Route::get('agents', [FeedbackAgentGroupController::class, 'index'])->name('agents.index'); Route::post('agents', [FeedbackAgentGroupController::class, 'store'])->name('agents.store'); Route::delete('agents/{item}', [FeedbackAgentGroupController::class, 'destroy'])->name('agents.destroy');
    });
});

Route::middleware(['web', 'auth'])->prefix('complaint')->group(function () {
    Route::get('/', [FeedbackTicketController::class, 'index'])->name('complaint.index');
    Route::get('create', [FeedbackTicketController::class, 'create'])->name('complaint.create');
    Route::post('/', [FeedbackTicketController::class, 'store'])->name('complaint.store');
    Route::get('{ticket}', [FeedbackTicketController::class, 'show'])->name('complaint.show');
    Route::get('{ticket}/edit', [FeedbackTicketController::class, 'edit'])->name('complaint.edit');
    Route::put('{ticket}', [FeedbackTicketController::class, 'update'])->name('complaint.update');
    Route::delete('{ticket}', [FeedbackTicketController::class, 'destroy'])->name('complaint.destroy');
});
