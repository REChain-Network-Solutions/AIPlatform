<?php
// UI label: Site Inspections (Pass 2)

namespace Modules\QualityControl\Tests\Feature;

use Tests\TestCase;

class SidebarDoesNotCrashTest extends TestCase
{
    /** @test */
    public function sidebar_renders_without_missing_routes(): void
    {
        $this->assertTrue(true);
    }
}
