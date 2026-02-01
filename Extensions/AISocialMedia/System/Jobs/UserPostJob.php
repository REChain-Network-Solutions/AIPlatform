<?php

namespace App\Extensions\AISocialMedia\System\Jobs;

use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\AISocialMedia\System\Models\ScheduledPost;
use App\Extensions\AISocialMedia\System\Services\AutomationService;
use App\Extensions\AISocialMedia\System\Services\Contracts\BaseService;
use App\Helpers\Classes\ApiHelper;
use App\Models\Setting;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use OpenAI\Laravel\Facades\OpenAI as FacadesOpenAI;
use RuntimeException;

class UserPostJob implements ShouldQueue
{
    use Dispatchable;
    use InteractsWithQueue;
    use Queueable;
    use SerializesModels;

    protected ScheduledPost $post;

    public function __construct(ScheduledPost $post)
    {
        $this->post = $post;
    }

    public function handle(): void
    {
        $this->getService($this->post)
            ->share(
                $this->content($this->post)
            );
    }

    public function content(ScheduledPost $post): ?string
    {
        if ($post->auto_generate) {
            $setting = Setting::getCache();

            ApiHelper::setOpenAiKey();
            // create the completion
            $completion = FacadesOpenAI::chat()->create([
                'model'    => $setting->openai_default_model,
                'messages' => [
                    [
                        'role'    => 'user',
                        'content' => $this->post->prompt,
                    ],
                ],
            ]);

            $driver = Entity::driver(
                EntityEnum::tryFrom($setting->openai_default_model)
            )->forUser($this->post->user);

            $driver->input($completion->choices[0]->message->content)->calculateCredit();

            $driver->redirectIfNoCreditBalance();

            $driver->decreaseCredit();

            return $completion->choices[0]->message->content;
        }

        return $post->getAttribute('content');
    }

    public function getService(ScheduledPost $post): BaseService
    {
        $platform = AutomationService::find($post->platform);

        $service = $platform?->service;

        if ($service instanceof BaseService) {
            return $service->setPlatform($platform)->setPost($post);
        }

        throw new RuntimeException('Service not found');
    }
}
