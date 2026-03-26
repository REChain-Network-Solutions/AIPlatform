<?php

return [
    'permissions' => [
        'compliance.view'      => ['Admin','Compliance Officer','Auditor'],
        'compliance.create'    => ['Admin','Compliance Officer'],
        'compliance.update'    => ['Admin','Compliance Officer'],
        'compliance.signoff'   => ['Admin','Compliance Officer'],
        'compliance.export'    => ['Admin','Compliance Officer'],
        'compliance.logs.view' => ['Admin','Compliance Officer','Auditor'],
    ],
];
