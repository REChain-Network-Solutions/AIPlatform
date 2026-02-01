<?php

declare(strict_types=1);

namespace App\Extensions\Chatbot\System;

use App\Extensions\Chatbot\System\Http\Controllers\Api\ChatbotApplicationController;
use App\Extensions\Chatbot\System\Http\Controllers\Api\ChatbotFrameController;
use App\Extensions\Chatbot\System\Http\Controllers\AvatarController;
use App\Extensions\Chatbot\System\Http\Controllers\ChatbotController;
use App\Extensions\Chatbot\System\Http\Controllers\ChatbotMultiChannelController;
use App\Extensions\Chatbot\System\Http\Controllers\ChatbotTrainController;
use App\Extensions\Chatbot\System\Http\Middleware\LanguageMiddleware;
use App\Helpers\Classes\Helper;
use App\Http\Middleware\CheckTemplateTypeAndPlan;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;

class ChatbotServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        $this->registerConfig();
    }

    public function boot(Kernel $kernel): void
    {
        $this->registerTranslations()
            ->registerViews()
            ->registerRoutes()
            ->registerMigrations()
            ->publishAssets()
            ->registerCommand();

    }

    public function registerCommand(): static
    {
        if (Helper::appIsDemo()) {
            $this->commands([
                Console\Commands\ClearDemoModeCommand::class,
            ]);

            //            if ($this->app->runningInConsole()) {
            //                $this->app->booted(function () {
            //                    $schedule = $this->app->make(Schedule::class);
            //                    $schedule->command('app:clear-chatbot-demo-mode')->everyMinute();
            //                });
            //            }
        }

        return $this;
    }

    public function publishAssets(): static
    {
        $this->publishes([
            __DIR__ . '/../resources/assets/js'     => public_path('vendor/chatbot/js'),
            __DIR__ . '/../resources/assets/images' => public_path('vendor/chatbot/images'),
            __DIR__ . '/../resources/assets/icons'  => public_path('vendor/chatbot/icons'),
        ], 'extension');

        return $this;
    }

    public function registerConfig(): static
    {
        $this->mergeConfigFrom(__DIR__ . '/../config/chatbot.php', 'chatbot');

        return $this;
    }

    protected function registerTranslations(): static
    {
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'chatbot');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'chatbot');

        return $this;
    }

    public function registerMigrations(): static
    {
        $this->loadMigrationsFrom(__DIR__ . '/../database/migrations');

        return $this;
    }

    private function registerRoutes(): static
    {
        $this->router()
            ->group([
                'middleware' => 'web',
            ], function (Router $router) {
                $router
                    ->controller(ChatbotFrameController::class)
                    ->group(function (Router $router) {
                        $router->get('chatbot/{chatbot:uuid}/frame', 'frame')->name('chatbot.frame');
                    });
            })
            ->group([
                'middleware'     => ['api', LanguageMiddleware::class],
                'prefix'         => 'api/v2/chatbot',
                'as'             => 'api.v2.chatbot.',
                'controller'     => ChatbotApplicationController::class,
            ], function (Router $router) {
                $router->get('{chatbot:uuid}', 'index')->name('index');
                $router->get('{chatbot:uuid}/session/{sessionId}', 'indexSession')->name('index.session');
                $router->post('{chatbot:uuid}/session/{sessionId}/conversation', 'conversionStore')->name('conversion.store');
                $router->post('{chatbot:uuid}/session/{sessionId}/conversation/connect', 'connectSupport')->name('conversion.connect.support');
                $router->get('{chatbot:uuid}/session/{sessionId}/conversation/{chatbotConversation}', 'conversion')->name('conversion.show');
                $router->get('{chatbot:uuid}/session/{sessionId}/conversation/{chatbotConversation}/messages', 'messages')->name('conversion.messages');
                $router->post('{chatbot:uuid}/session/{sessionId}/conversation/{chatbotConversation}/messages', 'storeMessage')->name('conversion.store.message');
            })
            ->group([
                'middleware' => ['web', 'auth'],
            ], function (Router $route) {
                $route->controller(ChatbotMultiChannelController::class)
                    ->name('dashboard.chatbot-multi-channel.')
                    ->prefix('dashboard/chatbot-multi-channel')
                    ->group(function () {
                        Route::any('', 'index')->name('index');
                        Route::POST('delete', 'delete')->name('delete');
                    });
                $route
                    ->controller(ChatbotController::class)
                    ->prefix('dashboard/chatbot')
                    ->name('dashboard.chatbot.')
                    ->group(function (Router $route) {
                        $route->get('', 'index')->name('index')->middleware(CheckTemplateTypeAndPlan::class);
                        $route->post('', 'store')->name('store');
                        $route->post('update', 'update')->name('update');
                        $route->delete('delete', 'delete')->name('delete');

                        // conversation
                        $route->get('conversations', 'conversations')->name('conversations');
                        $route->get('conversations-with-paginate', 'conversationsWithPaginate')->name('conversations.with.paginate');

                        // ended routes
                        $route->get('{chatbot}/enbed', 'enbed')->name('enbed');
                    });
                $route
                    ->controller(ChatbotTrainController::class)
                    ->prefix('dashboard/chatbot/train')
                    ->name('dashboard.chatbot.train.')
                    ->group(function (Router $route) {
                        // train routes
                        $route->get('data', 'trainData')->name('data');
                        $route->post('delete-embedding', 'deleteEmbedding')->name('delete');
                        $route->post('generate-embedding', 'generateEmbedding')->name('generate.embedding');
                        $route->get('{chatbot}', 'train')->name('index');
                        $route->post('url', 'trainUrl')->name('url');
                        $route->post('file', 'trainFile')->name('file');
                        $route->post('text', 'trainText')->name('text');
                        $route->post('qa', 'trainQa')->name('qa');
                    });
                $route->post('dashboard/chatbot/avatar/upload', AvatarController::class)
                    ->name('dashboard.chatbot.upload.avatar');
            });

        return $this;
    }

    private function router(): Router|Route
    {
        return $this->app['router'];
    }
}
