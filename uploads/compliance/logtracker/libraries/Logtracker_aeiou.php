<?php

defined('BASEPATH') || exit('No direct script access allowed');
require_once __DIR__.'/../vendor/autoload.php';
require_once __DIR__.'/../third_party/node.php';
use Firebase\JWT\JWT as logtracker_JWT;
use Firebase\JWT\Key as logtracker_Key;
use WpOrg\Requests\Requests as logtracker_Requests;

class Logtracker_aeiou
{
    public static function getPurchaseData($code)
    {
    }

    public static function verifyPurchase($code)
    {
    }

    public function validatePurchase($module_name)
    {
    }
}
