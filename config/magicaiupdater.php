<?php

use App\Services\UpdateChannel;

$beta = UpdateChannel::isBeta();

return [

    'tmp_folder_name' => 'tmp',

    'script_filename' => 'upgrade.php',

    'update_baseurl' => 'https://api.liquid-themes.com/magicai/updater-v2',

    'version_baseurl' => 'https://api.liquid-themes.com/magicai/updater-v2/magicaiupdater.json',

    'update_new_base_url' => $beta ? UpdateChannel::BETA_BASE_V2 : UpdateChannel::STABLE_BASE_V2,

    'version_new_base_url' => $beta ? UpdateChannel::BETA_BASE_V2 . '/magicaiupdater.json' : UpdateChannel::STABLE_BASE_V2 . '/magicaiupdater.json',
];
