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

namespace Tests\YooKassa\Request\PersonalData;

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\Metadata;
use YooKassa\Model\PersonalData\PersonalDataType;
use YooKassa\Request\PersonalData\CreatePayoutStatementRecipientPersonalDataRequestBuilder;
use YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest;
use YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest;
use YooKassa\Request\PersonalData\CreateSbpPayoutRecipientPersonalDataRequestBuilder;

/**
 * CreateSbpPayoutRecipientPersonalDataRequestBuilderTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreateSbpPayoutRecipientPersonalDataRequestBuilderTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testLastName(mixed $options): void
    {
        $builder = new CreateSbpPayoutRecipientPersonalDataRequestBuilder();
        $builder->setOptions($options);
        $instance = $builder->build();

        if (empty($options['last_name'])) {
            self::assertNull($instance->getLastName());
        } else {
            self::assertNotNull($instance->getLastName());
            self::assertEquals($options['last_name'], $instance->getLastName());
        }
    }

    /**
     * @dataProvider invalidLastNameDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidLastName(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $instance = new CreateSbpPayoutRecipientPersonalDataRequestBuilder();
        $instance->setLastName($value);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testFirstName(mixed $options): void
    {
        $builder = new CreateSbpPayoutRecipientPersonalDataRequestBuilder();
        $builder->setOptions($options);
        $instance = $builder->build();

        if (empty($options['first_name'])) {
            self::assertNull($instance->getFirstName());
        } else {
            self::assertNotNull($instance->getFirstName());
            self::assertEquals($options['first_name'], $instance->getFirstName());
        }
    }

    /**
     * @dataProvider invalidFirstNameDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidFirstName(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $instance = new CreateSbpPayoutRecipientPersonalDataRequestBuilder();
        $instance->setFirstName($value);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testMiddleName(mixed $options): void
    {
        $builder = new CreateSbpPayoutRecipientPersonalDataRequestBuilder();
        $builder->setOptions($options);
        $instance = $builder->build();

        if (empty($options['middle_name'])) {
            self::assertNull($instance->getMiddleName());
        } else {
            self::assertNotNull($instance->getMiddleName());
            self::assertEquals($options['middle_name'], $instance->getMiddleName());
        }
    }

    /**
     * @dataProvider invalidMiddleNameDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidMiddleName(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $instance = new CreateSbpPayoutRecipientPersonalDataRequestBuilder();
        $instance->setMiddleName($value);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testMetadata(mixed $options): void
    {
        $builder = new CreateSbpPayoutRecipientPersonalDataRequestBuilder();
        $builder->setOptions($options);
        $instance = $builder->build();

        if (empty($options['metadata'])) {
            self::assertNull($instance->getMetadata());
        } else {
            self::assertNotNull($instance->getMetadata());
            self::assertEquals($options['metadata'], $instance->getMetadata()->toArray());
        }
    }

    /**
     * @dataProvider invalidMetadataDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidMetadata(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $instance = new CreateSbpPayoutRecipientPersonalDataRequestBuilder();
        $instance->setMetadata($value);
    }

    public function testBuilder(): void
    {
        $builder = SbpPayoutRecipientPersonalDataRequest::builder();
        self::assertInstanceOf(CreateSbpPayoutRecipientPersonalDataRequestBuilder::class, $builder);
    }

    public static function validDataProvider(): array
    {
        $result = [
            [
                [
                    'type' => PersonalDataType::SBP_PAYOUT_RECIPIENT,
                    'first_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_FIRST_NAME, 'abcdefghijklmnopqrstuvwxyz'),
                    'last_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_LAST_NAME, 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'),
                    'middle_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_LAST_NAME, '- abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя'),
                    'metadata' => [Random::str(3, 128, 'abcdefghijklmnopqrstuvwxyz') => Random::str(1, 512)],
                ],
            ],
        ];
        for ($i = 0; $i < 10; $i++) {
            $request = [
                'type' => PersonalDataType::SBP_PAYOUT_RECIPIENT,
                'first_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_FIRST_NAME, 'abcdefghijklmnopqrstuvwxyz'),
                'last_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_LAST_NAME, 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'),
                'middle_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_LAST_NAME, '- abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя'),
                'metadata' => [Random::str(3, 128, 'abcdefghijklmnopqrstuvwxyz') => Random::str(1, 512)],
            ];
            $result[] = [$request];
        }

        return $result;
    }

    public static function invalidTypeDataProvider(): array
    {
        return [
            [false],
            [true],
            [Random::str(10)],
        ];
    }

    public static function invalidMetadataDataProvider(): array
    {
        return [
            [false],
            [true],
            [1],
            [Random::str(10)],
        ];
    }

    public static function invalidLastNameDataProvider(): array
    {
        return [
            [''],
            [false],
            [Random::str(AbstractPersonalDataRequest::MAX_LENGTH_FIRST_NAME + 1)],
        ];
    }

    public static function invalidMiddleNameDataProvider(): array
    {
        return [
            [Random::str(AbstractPersonalDataRequest::MAX_LENGTH_LAST_NAME + 1)],
        ];
    }

    public static function invalidFirstNameDataProvider(): array
    {
        return [
            [''],
            [false],
            [Random::str(AbstractPersonalDataRequest::MAX_LENGTH_FIRST_NAME + 1)],
        ];
    }
}
