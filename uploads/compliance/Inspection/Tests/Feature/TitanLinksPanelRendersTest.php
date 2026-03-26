<?php

namespace Modules\Inspection\Tests\Feature;

use Tests\TestCase;

class TitanLinksPanelRendersTest extends TestCase
{
    public function test_panel_view_exists(): void
    {
        $this->assertTrue(view()->exists('inspection::partials.titan-links'));
    }
}
