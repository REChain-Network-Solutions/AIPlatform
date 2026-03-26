<?php

namespace Modules\Inspection\Tests\Feature;

use Illuminate\Support\Facades\Route;
use Tests\TestCase;

class TemplatesRoutesTest extends TestCase
{
    public function test_templates_routes_are_registered(): void
    {
        $this->assertTrue(Route::has('inspection-templates.index'));
    }
}
