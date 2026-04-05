<?php
namespace App\View\Components\CustomModules;

use Illuminate\View\Component;
use Illuminate\View\View;

class QuickStat extends Component
{
    public function __construct(public string $label, public string $value) {}

    public function render(): View
    {
        return view('components.custom-modules.quick-stat');
    }
}
