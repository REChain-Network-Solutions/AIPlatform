<?php
// UI label: Site Inspections (Pass 2)

namespace Modules\QualityControl\Tests\Unit;

use Tests\TestCase;

class SupportClassesTest extends TestCase
{
    /** @test */
    public function constants_load(): void
    {
        $this->assertTrue(class_exists(\Modules\QualityControl\Support\InspectionRoutes::class));
    }
}
