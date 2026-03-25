<?php

namespace Modules\EInvoice\Views\Components\Form;

use Illuminate\View\Component;

class Setting extends Component
{

    public $setting;

    /**
     * Create a new component instance.
     *
     * @return void
     */
    public function __construct()
    {
        $this->setting = \Modules\EInvoice\Entities\EInvoiceCompanySetting::first();
    }

    /**
     * Get the view / contents that represent the component.
     *
     * @return \Illuminate\View\View|string
     */
    public function render()
    {
        return view('einvoice::components.form.setting');
    }

}
