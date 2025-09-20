<?php

/*
* The MIT License
*
* Copyright (c) 2025 "YooMoney", NBСO LLC
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

namespace Tests\YooKassa\Request\Deals;

use Exception;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\Deal\DealStatus;
use YooKassa\Model\Deal\SafeDeal;
use YooKassa\Request\Deals\DealsRequest;
use YooKassa\Request\Deals\DealsRequestBuilder;

/**
 * DealsRequestBuilderTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class DealsRequestBuilderTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetCursor(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCursor());

        $builder->setCursor($options['cursor']);
        $instance = $builder->build();
        if (empty($options['cursor'])) {
            self::assertNull($instance->getCursor());
        } else {
            self::assertEquals($options['cursor'], $instance->getCursor());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetCreatedAtGte(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCreatedAtGte());

        $builder->setCreatedAtGte($options['createdAtGte'] ?? null);
        $instance = $builder->build();
        if (empty($options['createdAtGte'])) {
            self::assertNull($instance->getCreatedAtGte());
        } else {
            self::assertEquals($options['createdAtGte'], $instance->getCreatedAtGte()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetCreatedGt(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCreatedAtGt());

        $builder->setCreatedAtGt($options['createdAtGt'] ?? null);
        $instance = $builder->build();
        if (empty($options['createdAtGt'])) {
            self::assertNull($instance->getCreatedAtGt());
        } else {
            self::assertEquals($options['createdAtGt'], $instance->getCreatedAtGt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetCreatedLte(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCreatedAtLte());

        $builder->setCreatedAtLte($options['createdAtLte'] ?? null);
        $instance = $builder->build();
        if (empty($options['createdAtLte'])) {
            self::assertNull($instance->getCreatedAtLte());
        } else {
            self::assertEquals($options['createdAtLte'], $instance->getCreatedAtLte()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetCreatedLt(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCreatedAtLt());

        $builder->setCreatedAtLt($options['createdAtLt'] ?? null);
        $instance = $builder->build();
        if (empty($options['createdAtLt'])) {
            self::assertNull($instance->getCreatedAtLt());
        } else {
            self::assertEquals($options['createdAtLt'], $instance->getCreatedAtLt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetExpiresAtGte(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getExpiresAtGte());

        $builder->setExpiresAtGte($options['expiresAtGte'] ?? null);
        $instance = $builder->build();
        if (empty($options['expiresAtGte'])) {
            self::assertNull($instance->getExpiresAtGte());
        } else {
            self::assertEquals($options['expiresAtGte'], $instance->getExpiresAtGte()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetExpiresGt(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getExpiresAtGt());

        $builder->setExpiresAtGt($options['expiresAtGt'] ?? null);
        $instance = $builder->build();
        if (empty($options['expiresAtGt'])) {
            self::assertNull($instance->getExpiresAtGt());
        } else {
            self::assertEquals($options['expiresAtGt'], $instance->getExpiresAtGt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetExpiresLte(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getExpiresAtLte());

        $builder->setExpiresAtLte($options['expiresAtLte'] ?? null);
        $instance = $builder->build();
        if (empty($options['expiresAtLte'])) {
            self::assertNull($instance->getExpiresAtLte());
        } else {
            self::assertEquals($options['expiresAtLte'], $instance->getExpiresAtLte()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetExpiresLt(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getExpiresAtLt());

        $builder->setExpiresAtLt($options['expiresAtLt'] ?? null);
        $instance = $builder->build();
        if (empty($options['expiresAtLt'])) {
            self::assertNull($instance->getExpiresAtLt());
        } else {
            self::assertEquals($options['expiresAtLt'], $instance->getExpiresAtLt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetFullTextSearch(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getFullTextSearch());

        $builder->setFullTextSearch($options['fullTextSearch'] ?? null);
        $instance = $builder->build();
        if (empty($options['fullTextSearch'])) {
            self::assertNull($instance->getFullTextSearch());
        } else {
            self::assertEquals($options['fullTextSearch'], $instance->getFullTextSearch());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetLimit(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNotNull($instance->getLimit());

        $builder->setLimit($options['limit']);
        $instance = $builder->build();
        if (is_null($options['limit'])) {
            self::assertNull($instance->getLimit());
        } else {
            self::assertEquals($options['limit'], $instance->getLimit());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetStatus(mixed $options): void
    {
        $builder = new DealsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getStatus());

        $builder->setStatus($options['status']);
        $instance = $builder->build();
        if (empty($options['status'])) {
            self::assertNull($instance->getStatus());
        } else {
            self::assertEquals($options['status'], $instance->getStatus());
        }
    }

    public function validDataProvider(): array
    {
        $result = [
            [
                [
                    'createdAtGte' => null,
                    'createdAtGt' => null,
                    'createdAtLte' => null,
                    'createdAtLt' => null,
                    'expiresAtGte' => null,
                    'expiresAtGt' => null,
                    'expiresAtLte' => null,
                    'expiresAtLt' => null,
                    'fullTextSearch' => null,
                    'status' => null,
                    'limit' => Random::int(1, DealsRequest::MAX_LIMIT_VALUE),
                    'cursor' => null,
                ],
            ],
            [
                [
                    'fullTextSearch' => '',
                    'status' => '',
                    'limit' => 1,
                    'cursor' => '',
                ],
            ],
        ];
        $statuses = DealStatus::getValidValues();
        for ($i = 0; $i < 10; $i++) {
            $request = [
                'createdAtGte' => date(YOOKASSA_DATE, Random::int(1, time())),
                'createdAtGt' => date(YOOKASSA_DATE, Random::int(1, time())),
                'createdAtLte' => date(YOOKASSA_DATE, Random::int(1, time())),
                'createdAtLt' => date(YOOKASSA_DATE, Random::int(1, time())),
                'expiresAtGte' => date(YOOKASSA_DATE, Random::int(1, time())),
                'expiresAtGt' => date(YOOKASSA_DATE, Random::int(1, time())),
                'expiresAtLte' => date(YOOKASSA_DATE, Random::int(1, time())),
                'expiresAtLt' => date(YOOKASSA_DATE, Random::int(1, time())),
                'fullTextSearch' => Random::str(DealsRequest::MIN_LENGTH_DESCRIPTION, SafeDeal::MAX_LENGTH_DESCRIPTION),
                'status' => $statuses[Random::int(0, count($statuses) - 1)],
                'limit' => Random::int(1, DealsRequest::MAX_LIMIT_VALUE),
                'cursor' => $this->randomString(Random::int(2, 30)),
            ];
            $result[] = [$request];
        }

        return $result;
    }

    private function randomString($length, $any = true): string
    {
        static $chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-+_.';

        $result = '';
        for ($i = 0; $i < $length; $i++) {
            if ($any) {
                $char = chr(Random::int(32, 126));
            } else {
                $rnd = Random::int(0, strlen($chars) - 1);
                $char = $chars[$rnd];
            }
            $result .= $char;
        }

        return $result;
    }
}
