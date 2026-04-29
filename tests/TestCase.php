<?php

declare(strict_types=1);

namespace Tests;

use Database\Seeders\DatabaseSeeder;
use Illuminate\Foundation\Testing\TestCase as BaseTestCase;
use Spatie\Once\Cache as OnceCache;

abstract class TestCase extends BaseTestCase
{
    use CreatesApplication;

    // protected bool $seed = true;
    //
    // protected string $seeder = DatabaseSeeder::class;

    protected function setUp(): void
    {
        parent::setUp();

        OnceCache::getInstance()->disable();

        $this->usePerProcessCompiledViews();
    }

    /**
     * Give each parallel test process its own compiled views directory
     * to prevent race conditions on shared Blade cache files.
     */
    protected function usePerProcessCompiledViews(): void
    {
        $token = getenv('TEST_TOKEN');

        if ($token === false) {
            return;
        }

        $path = storage_path('framework/views/' . $token);

        if (! is_dir($path)) {
            mkdir($path, 0755, true);
        }

        config(['view.compiled' => $path]);
    }
}
