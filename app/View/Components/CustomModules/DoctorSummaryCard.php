<?php
namespace App\View\Components\CustomModules;

use Illuminate\View\Component;
use Illuminate\View\View;

class DoctorSummaryCard extends Component
{
    public function __construct(
        public string $title,
        public string $body,
        public string $theme = 'primary',
    ) {}

    public function render(): View
    {
        return view('components.custom-modules.doctor-summary-card');
    }
}
