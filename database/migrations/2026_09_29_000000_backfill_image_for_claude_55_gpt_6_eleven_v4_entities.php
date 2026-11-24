<?php

use App\Domains\Entity\Models\Entity;
use App\Enums\StatusEnum;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    /**
     * IMPORTANT: use raw string keys/titles here, never EntityEnum cases.
     * Migrations must keep running on old installs whose codebase predates the
     * enum cases — referencing EntityEnum::X would fatal with an undefined
     * case error there. EntitySeeder (which uses the enum) covers fresh
     * installs; this migration covers existing rows.
     *
     * @var array<int, array{key: string, engine: string, title: string, logo: ?string}>
     */
    private array $entities;

    public function __construct()
    {
        $claude = 'upload/enginelogo/claude_logo.svg';
        $openai = 'upload/enginelogo/openai.svg';

        $this->entities = [
            ['key' => 'claude-opus-5-5', 'engine' => 'anthropic', 'title' => 'Claude Opus 5.5', 'logo' => $claude],
            ['key' => 'claude-sonnet-5-5', 'engine' => 'anthropic', 'title' => 'Claude Sonnet 5.5', 'logo' => $claude],
            ['key' => 'gpt-6-luna', 'engine' => 'openai', 'title' => 'GPT-6 Luna (May 18, 2026 knowledge cutoff, 128k max output tokens, 1.05M context.)', 'logo' => $openai],
            ['key' => 'gpt-6-sol', 'engine' => 'openai', 'title' => 'GPT-6 Sol (Apr 20, 2026 knowledge cutoff, 128k max output tokens, 1.05M context.)', 'logo' => $openai],
            ['key' => 'eleven_v4', 'engine' => 'elevenlabs', 'title' => 'ElevenLabs v4 for TTS', 'logo' => null],
            ['key' => 'eleven_v4_turbo', 'engine' => 'elevenlabs', 'title' => 'ElevenLabs v4 Turbo for TTS', 'logo' => null],
        ];
    }

    public function up(): void
    {
        $elevenlabsLogo = $this->elevenlabsLogo();

        foreach ($this->entities as $entity) {
            if (str_starts_with($entity['key'], 'eleven_')) {
                $entity['logo'] = $elevenlabsLogo;
            }

            Entity::query()->updateOrCreate(
                ['key' => $entity['key']],
                [
                    'engine' => $entity['engine'],
                    'title'  => $entity['title'],
                    'status' => StatusEnum::ENABLED->value,
                    'image'  => $entity['logo'],
                ]
            );
        }
    }

    public function down(): void
    {
        foreach ($this->entities as $entity) {
            if ($entity['logo'] === null) {
                continue;
            }

            Entity::query()
                ->where('key', $entity['key'])
                ->where('image', $entity['logo'])
                ->update(['image' => null]);
        }
    }

    /**
     * There is no dedicated ElevenLabs logo asset, so inherit whatever image the
     * existing ElevenLabs rows carry (null on fresh installs, where EntitySeeder
     * creates rows without an image).
     */
    private function elevenlabsLogo(): ?string
    {
        try {
            return Entity::query()->where('key', 'elevenlabs')->value('image');
        } catch (Throwable) {
            return null;
        }
    }
};
