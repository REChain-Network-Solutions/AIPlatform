<?php

namespace Tests\YooKassa\Helpers\Config;

use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Config\ConfigurationLoader;

/**
 * @internal
 */
class ConfigurationLoaderTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $fileName
     */
    public function testLoad($fileName): void
    {
        $loader = new ConfigurationLoader();
        $loader->load($fileName);
        if (empty($fileName)) {
            $fileName = __DIR__ . DIRECTORY_SEPARATOR . '..' . DIRECTORY_SEPARATOR . '..' . DIRECTORY_SEPARATOR . '..' . DIRECTORY_SEPARATOR . 'lib' . DIRECTORY_SEPARATOR . 'configuration.json';
        }
        $data = file_get_contents($fileName);
        self::assertEquals(json_decode($data, true), $loader->getConfig());
    }

    public static function validDataProvider()
    {
        return [
            [null],
            [''],
            [__DIR__ . DIRECTORY_SEPARATOR . 'test_config.json'],
        ];
    }
}
