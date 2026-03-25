<?php

$addOnOf = 'worksuite-saas-new';

return [
    'name' => 'EInvoice',
    'verification_required' => true,
    'envato_item_id' => 49301582,
    'parent_envato_id' => 23263417,
    'parent_min_version' => '5.3.54',
    'script_name' => $addOnOf.'-einvoice-module',
    'parent_product_name' => $addOnOf,
    'setting' => \Modules\EInvoice\Entities\EInvoiceSetting::class,
];

