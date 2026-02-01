<?php

declare(strict_types=1);

namespace App\Extensions\ChatSetting\System;

use App\Extensions\ChatSetting\System\Http\Controllers\Chatbot\ChatbotController;
use App\Extensions\ChatSetting\System\Http\Controllers\Chatbot\ChatbotTrainingController;
use App\Extensions\ChatSetting\System\Http\Controllers\ChatCategoryController;
use App\Extensions\ChatSetting\System\Http\Controllers\ChatTemplateController;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;

class ChatSettingServiceProvider extends ServiceProvider
{
    public function register(): void {}

    public function boot(Kernel $kernel): void
    {
        $this->registerViews()
            ->registerRoutes();

    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'chat-setting');

        return $this;
    }

    private function registerRoutes(): void
    {
        $this->router()
            ->group([
                'middleware' => ['web', 'auth'],
            ], function (Router $router) {
                $router
                    ->prefix('dashboard/user/chat-setting')
                    ->name('dashboard.user.chat-setting.')
                    ->group(function (Router $router) {
                        $router->resource('chat-category', ChatCategoryController::class)

                            ->except('show', 'destroy');
                        Route::get('chat-category/{chat_category}/delete', [ChatCategoryController::class, 'destroy'])
                            ->name('chat-category.destroy');

                        Route::resource('chat-template', ChatTemplateController::class)
                            ->except('show');

                        Route::group([
                            'as'         => 'chatbot.',
                            'prefix'     => 'chatbot/{chatbot}',
                            'controller' => ChatbotTrainingController::class,
                        ], static function () {
                            Route::post('text', 'text')->name('text');
                            Route::post('qa', 'qa')->name('qa');

                            Route::post('training', 'training')->name('training');
                            Route::get('web-sites', 'getWebSites')->name('web-sites');
                            Route::post('web-sites', 'postWebSites');
                            Route::post('upload-pdf', 'uploadPdf')->name('upload-pdf');
                            Route::delete('item/{id}', 'deleteItem')->name('item.delete');
                        });

                        Route::resource('chatbot', ChatbotController::class);
                    });

            });
    }

    private function router(): Router|Route
    {
        return $this->app['router'];
    }
}
