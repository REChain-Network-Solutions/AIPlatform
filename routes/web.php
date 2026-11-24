<?php

declare(strict_types=1);

use App\Http\Controllers\BlogController;
use App\Http\Controllers\Common\CheckSubscriptionEndController;
use App\Http\Controllers\Common\LocaleController;
use App\Http\Controllers\Common\SitemapController;
use App\Http\Controllers\Common\SystemSlotController;
use App\Http\Controllers\IndexController;
use App\Http\Controllers\MailController;
use App\Http\Controllers\PageController;
use App\Http\Controllers\Payment\PlanAndPricingController;
use App\Http\Controllers\PrivatePlanController;
use App\Http\Controllers\SharedCreditPreviewController;
use Illuminate\Support\Facades\Route;

Route::get('language/{lang}/change', LocaleController::class)->name('language.change');

Route::get('default', static function () {
    return response()->noContent(

    );
})->name('default');
Route::view('account-deletion', 'default.account-deletion');
Route::middleware('checkInstallation')
    ->group(static function () {
        Route::get('', IndexController::class)->name('index');
        Route::controller(PageController::class)
            ->group(static function () {
                Route::get('privacy-policy', 'pagePrivacy')->name('pagePrivacy');
                Route::get('terms', 'pageTerms')->name('pageTerms');
                Route::get('page/{slug}', 'pageContent')->name('pageContent');
            });

        Route::controller(BlogController::class)
            ->group(static function () {
                Route::get('blog', 'index')->name('blog.index');
                Route::get('blog/{slug}', 'post')->name('blog.post');
                Route::get('blog/tag/{slug}', 'tags')->name('blog.tags');
                Route::get('blog/category/{slug}', 'categories')->name('blog.categories');
                Route::get('blog/author/{slug}', 'author')->name('blog.author');
            });

        Route::get('credit-list-partial', [PlanAndPricingController::class, 'creditListPartial'])->name('credit-list-partial');
        Route::get('team-credit-list-partial', [PlanAndPricingController::class, 'teamCreditListPartial'])->name('team-credit-list-partial');
    });

Route::get('sitemap.xml', [SitemapController::class, 'index']);
Route::get('plan/private/subscription/{key}', [PrivatePlanController::class, 'index']);
Route::get('confirm/email/{email_confirmation_code}', [MailController::class, 'emailConfirmationMail']);

Route::get('sys/{tk}', [SystemSlotController::class, 'index'])->middleware(['auth', 'admin', 'throttle:10,1']);
Route::post('sys/{slot}', [SystemSlotController::class, 'record'])->middleware(['auth', 'admin', 'throttle:10,1']);
Route::get('check-subscription-end', CheckSubscriptionEndController::class)->name('check-subscription-end');

Route::get('shared-credit/balance', [SharedCreditPreviewController::class, 'balance'])
    ->middleware('auth')
    ->name('shared-credit.balance');

Route::post('shared-credit/cost-preview', [SharedCreditPreviewController::class, 'preview'])
    ->middleware('auth')
    ->name('shared-credit.cost-preview');

if (file_exists(base_path('routes/custom_routes_web.php'))) {
    include base_path('routes/custom_routes_web.php');
}

require __DIR__ . '/auth.php';
require __DIR__ . '/panel.php';
require __DIR__ . '/webhooks.php';
