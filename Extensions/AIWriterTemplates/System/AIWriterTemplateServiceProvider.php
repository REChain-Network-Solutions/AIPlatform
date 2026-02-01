<?php

declare(strict_types=1);

namespace App\Extensions\AIWriterTemplates\System;

use App\Domains\Marketplace\Contracts\UninstallExtensionServiceProviderInterface;
use App\Models\OpenAIGenerator;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\ServiceProvider;

class AIWriterTemplateServiceProvider extends ServiceProvider implements UninstallExtensionServiceProviderInterface
{
    public function register(): void {}

    public function boot(Kernel $kernel): void
    {
        $this->loadMigrationsFrom(__DIR__ . '/../database/migrations');
    }

    public static function uninstall(): void
    {
        $openai = Storage::disk('extension')->get('AIWriterTemplates/database/data/openai.json');

        $data = json_decode($openai, true);

        foreach ($data as $item) {
            OpenaiGenerator::query()->where('slug', $item['slug'])->delete();
        }

        DB::table('migrations')->where('migration', '=', '2024_07_11_073517_ai_writer_templates_migrations')->delete();
    }
}
