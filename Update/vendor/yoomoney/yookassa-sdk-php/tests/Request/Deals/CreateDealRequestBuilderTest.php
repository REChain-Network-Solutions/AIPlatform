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
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use stdClass;
use YooKassa\Helpers\Random;
use YooKassa\Model\Deal\DealType;
use YooKassa\Model\Deal\FeeMoment;
use YooKassa\Model\Deal\SafeDeal;
use YooKassa\Model\Metadata;
use YooKassa\Request\Deals\CreateDealRequestBuilder;

/**
 * CreateDealRequestBuilderTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreateDealRequestBuilderTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetType(mixed $options): void
    {
        $builder = new CreateDealRequestBuilder();

        $builder->setType($options['type']);
        $instance = $builder->build($this->getRequiredData('type'));

        if (null === $options['type'] || '' === $options['type']) {
            self::assertNull($instance->getType());
        } else {
            self::assertNotNull($instance->getType());
            self::assertEquals($options['type'], $instance->getType());
        }
    }

    /**
     * @dataProvider invalidDescriptionDataProvider
     *
     * @param mixed $options
     */
    public function testSetInvalidType(mixed $options): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreateDealRequestBuilder();
        $builder->setType($options);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetFeeMoment(mixed $options): void
    {
        $builder = new CreateDealRequestBuilder();

        $builder->setFeeMoment($options['fee_moment']);
        $instance = $builder->build($this->getRequiredData('fee_moment'));

        if (null === $options['fee_moment'] || '' === $options['fee_moment']) {
            self::assertNull($instance->getFeeMoment());
        } else {
            self::assertNotNull($instance->getFeeMoment());
            self::assertEquals($options['fee_moment'], $instance->getFeeMoment());
        }
    }

    /**
     * @dataProvider invalidDescriptionDataProvider
     *
     * @param mixed $options
     */
    public function testSetInvalidFeeMoment(mixed $options): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreateDealRequestBuilder();
        $builder->setFeeMoment($options);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetMetadata(mixed $options): void
    {
        $builder = new CreateDealRequestBuilder();

        $instance = $builder->build($this->getRequiredData());
        self::assertNull($instance->getMetadata());

        $builder->setMetadata($options['metadata']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['metadata'])) {
            self::assertNull($instance->getMetadata());
        } else {
            self::assertEquals($options['metadata'], $instance->getMetadata()->toArray());
        }
    }

    /**
     * @dataProvider invalidDataProvider
     *
     * @param mixed $options
     */
    public function testSetInvalidMetadata(mixed $options): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreateDealRequestBuilder();
        $builder->setMetadata($options);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetDescription(mixed $options): void
    {
        $builder = new CreateDealRequestBuilder();

        $builder->setDescription($options['description']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['description'])) {
            self::assertNull($instance->getDescription());
        } else {
            self::assertEquals($options['description'], $instance->getDescription());
        }
    }

    /**
     * @dataProvider invalidDescriptionDataProvider
     *
     * @param mixed $options
     */
    public function testSetInvalidDescription(mixed $options): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreateDealRequestBuilder();
        $builder->setDescription($options);
    }

    /**
     * @return void
     * @throws Exception
     */
    public function testSetInvalidLengthDescription(): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreateDealRequestBuilder();
        $description = Random::str(SafeDeal::MAX_LENGTH_DESCRIPTION + 1);
        $builder->setDescription($description);
    }

    /**
     * @throws Exception
     */
    public static function validDataProvider(): array
    {
        $result = [
            [
                [
                    'type' => Random::value(DealType::getValidValues()),
                    'fee_moment' => Random::value(FeeMoment::getValidValues()),
                    'description' => null,
                    'metadata' => null,
                ],
            ],
            [
                [
                    'type' => Random::value(DealType::getValidValues()),
                    'fee_moment' => Random::value(FeeMoment::getValidValues()),
                    'description' => Random::str(1, SafeDeal::MAX_LENGTH_DESCRIPTION),
                    'metadata' => [new Metadata()],
                ],
            ],
        ];
        for ($i = 0; $i < 10; $i++) {
            $request = [
                'type' => Random::value(DealType::getValidValues()),
                'fee_moment' => Random::value(FeeMoment::getValidValues()),
                'description' => Random::str(1, SafeDeal::MAX_LENGTH_DESCRIPTION),
                'metadata' => [Random::str(1, 30) => Random::str(1, 128)],
            ];
            $result[] = [$request];
        }

        return $result;
    }

    /**
     * @return array
     */
    public static function invalidDataProvider(): array
    {
        return [
            [false],
            [true],
            [new stdClass()],
            [new SafeDeal()],
        ];
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public static function invalidDescriptionDataProvider(): array
    {
        return [
            [Random::str(SafeDeal::MAX_LENGTH_DESCRIPTION + 1)],
        ];
    }

    /**
     * @param null $testingProperty
     *
     * @throws Exception
     */
    protected function getRequiredData($testingProperty = null): array
    {
        $result = [];

        if ('type' !== $testingProperty) {
            $result['type'] = Random::value(DealType::getValidValues());
        }

        if ('fee_moment' !== $testingProperty) {
            $result['fee_moment'] = Random::value(FeeMoment::getValidValues());
        }

        return $result;
    }
}
