<?php

namespace Tests\YooKassa\Client;

use PHPUnit\Framework\TestCase;
use YooKassa\Client\ApiClientInterface;
use YooKassa\Client\BaseClient;
use YooKassa\Client\CurlClient;
use YooKassa\Helpers\Config\ConfigurationLoader;
use YooKassa\Helpers\Random;

/**
 * @internal
 */
class BaseClientTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $apiClient
     * @param mixed $configLoader
     */
    public function testSetAuth($apiClient, $configLoader): void
    {
        $instance = self::getInstance($apiClient, $configLoader);
        $instance->setAuth('123456', 'shopPassword');
        self::assertTrue($instance->getApiClient() instanceof ApiClientInterface);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $apiClient
     * @param mixed $configLoader
     */
    public function testSetAuthToken($apiClient, $configLoader): void
    {
        $instance = self::getInstance($apiClient, $configLoader);
        $instance->setAuthToken(Random::str(36));
        self::assertTrue($instance->getApiClient() instanceof ApiClientInterface);
    }

    /**
     * @dataProvider validConfigurationDataProvider
     *
     * @param mixed $value
     */
    public function testSetApiClient($value): void
    {
        $instance = self::getInstance();

        $client = new CurlClient();
        $client->setConnectionTimeout($value['connectionTimeout']);
        $client->setTimeout($value['timeout']);
        $client->setProxy($value['proxy']);
        $client->setConfig($value['config']);
        $client->setShopId($value['shopId']);
        $client->setShopPassword($value['shopPassword']);

        $instance->setApiClient($client);
        self::assertEquals($client->getConfig(), $instance->getConfig());
        self::assertTrue($instance->getApiClient() instanceof ApiClientInterface);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $apiClient
     * @param mixed $configLoader
     */
    public function testGetSetConfig($apiClient, $configLoader): void
    {
        $config = ['url' => 'test:url'];

        $instance = self::getInstance($apiClient, $configLoader);
        $instance->setConfig($config);

        $client = new CurlClient();
        $client->setConfig($config);
        self::assertEquals($client->getConfig(), $instance->getConfig());
    }

    /**
     * @dataProvider validIPv4DataProvider
     *
     * @param mixed $ip
     */
    public function testIsIPInTrustedRangeValid($ip): void
    {
        $instance = self::getInstance();

        $checkResult = $instance->isNotificationIPTrusted($ip);

        self::assertEquals(true, $checkResult);
    }

    /**
     * @dataProvider inValidIPv4DataProvider
     *
     * @param mixed $ip
     */
    public function testIsIPInTrustedRangeInValid($ip): void
    {
        $instance = self::getInstance();

        $checkResult = $instance->isNotificationIPTrusted($ip);

        self::assertEquals(false, $checkResult);
    }

    public static function validDataProvider(): array
    {
        return [
            [
                'apiClient' => null,
                'configLoader' => null,
            ],
            [
                'apiClient' => new CurlClient(),
                'configLoader' => new ConfigurationLoader(),
            ],
        ];
    }

    public static function validConfigurationDataProvider(): array
    {
        return [
            [
                [
                    'connectionTimeout' => 10,
                    'timeout' => 10,
                    'proxy' => 'proxy_url:8889',
                    'config' => ['url' => 'test:url'],
                    'shopId' => 'shopId',
                    'shopPassword' => 'shopPassword',
                ],
            ],
            [
                [
                    'connectionTimeout' => 30,
                    'timeout' => 30,
                    'proxy' => '123.456.789.5',
                    'config' => [],
                    'shopId' => null,
                    'shopPassword' => null,
                ],
            ],
        ];
    }

    public static function validIPv4DataProvider()
    {
        return [
            ['185.71.76.' . Random::int(1, 31)],
            ['185.71.77.' . Random::int(1, 31)],
            ['77.75.153.' . Random::int(1, 127)],
            ['77.75.154.' . Random::int(128, 254)],
            ['77.75.156.11'],
            ['77.75.156.35'],
            ['2a02:5180:0000:2669:0000:0000:0000:7d35'],
            ['2a02:5180:0000:2655:0000:0000:7d35:0000'],
            ['2a02:5180:0000:1533:0000:7d35:0000:0000'],
            ['2a02:5180:0000:2669:7d35:0000:0000:0000'],
        ];
    }

    public static function inValidIPv4DataProvider()
    {
        return [
            ['185.71.76.32'],
            ['185.71.77.32'],
            ['185.71.153.128'],
            ['185.71.154.' . Random::int(1, 128)],
            ['127.0.0.1'],
            ['77.75.156.12'],
            ['192.168.1.1'],
            ['8701:746f:d4f1:d39d:9dcc:6ea2:875e:7d35'],
            ['::1'],
        ];
    }

    /**
     * @param null $apiClient
     * @param null $configLoader
     */
    protected static function getInstance($apiClient = null, $configLoader = null): BaseClient
    {
        return new BaseClient($apiClient, $configLoader);
    }
}
