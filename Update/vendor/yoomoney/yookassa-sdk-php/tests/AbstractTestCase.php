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

namespace Tests\YooKassa;

use Exception;
use Faker\Factory;
use Faker\Generator;
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use ReflectionClass;
use ReflectionMethod;
use ReflectionNamedType;
use stdClass;
use TypeError;
use YooKassa\Common\AbstractEnum;
use YooKassa\Common\AbstractObject;
use YooKassa\Common\ListObject;
use YooKassa\Helpers\Random;
use YooKassa\Model\Metadata;
use YooKassa\Validator\Constraints\AllType;
use YooKassa\Validator\Constraints\Choice;
use YooKassa\Validator\Constraints\Count;
use YooKassa\Validator\Constraints\Date;
use YooKassa\Validator\Constraints\DateTime as DateTimeValidation;
use YooKassa\Validator\Constraints\Email;
use YooKassa\Validator\Constraints\GreaterThan;
use YooKassa\Validator\Constraints\GreaterThanOrEqual;
use YooKassa\Validator\Constraints\Ip;
use YooKassa\Validator\Constraints\Length;
use YooKassa\Validator\Constraints\LessThan;
use YooKassa\Validator\Constraints\LessThanOrEqual;
use YooKassa\Validator\Constraints\NotBlank;
use YooKassa\Validator\Constraints\NotNull;
use YooKassa\Validator\Constraints\Regex;
use YooKassa\Validator\Constraints\Type;
use YooKassa\Validator\Constraints\Url;
use YooKassa\Validator\Exceptions\EmptyPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Validator\Exceptions\ValidatorParameterException;
use YooKassa\Validator\Validator;
use DateTime;

/**
 * AbstractTestCase
 *
 * PHP version 8.0
 *
 * @category    ClassTest
 * @description Абстрактный класс по генерации валидных и невалидных данных для тестов.
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class AbstractTestCase extends TestCase
{
    /**
     * Возвращает валидный заполненный экземпляр класса на основе класса
     *
     * @param AbstractObject $class
     * @param bool $isArray
     *
     * @return array
     * @throws \ReflectionException
     */
    public function getValidDataProviderByClass(AbstractObject $class, bool $isArray = false): array
    {
        return [$this->prepareArrayForObject($class, $isArray)];
    }

    /**
     * Возвращает валидные данные на основе правил свойства
     *
     * @param array|null $propRules
     *
     * @return array
     * @throws \ReflectionException
     */
    public function getValidDataProviderByType(?array $propRules): array
    {
        if (empty($propRules)) {
            return [];
        }

        $arrayProps = $this->parsePropRules($propRules);
        if (empty($arrayProps) || !isset($arrayProps['type'])) {
            return [];
        }

        $result = [];
        if (!$arrayProps['isRequired']) {
            $result = [
                [null]
            ];

            if ($arrayProps['type'] !== 'float' && $arrayProps['type'] !== 'int') {
                $result[] = [''];
            }
        }

        if (isset($arrayProps['regex'])) {
            $max = Random::int(1, 10);
            for ($i = 0; $i < $max; $i++) {
                $result[] = [$this->prepareDataByRegex($arrayProps['regex'], $arrayProps['max'] ?? null, isset($arrayProps['min']) ?? null)];
            }
            return $result;
        }
        return array_merge($result, $this->prepareDataByType($arrayProps));
    }

    /**
     * Возвращает невалидные данные на основе правил свойства
     *
     * @param array|null $propRules
     *
     * @return array
     */
    public function getInvalidDataProviderByType(?array $propRules): array
    {
        if (empty($propRules)) {
            return [];
        }

        $arrayProps = $this->parsePropRules($propRules);
        if (empty($arrayProps)) {
            return [];
        }
        $result = [];

        if (isset($arrayProps['isRequired']) && $arrayProps['isRequired']) {
            $result = [
                [null, EmptyPropertyValueException::class]
            ];

            if ($arrayProps['type'] === 'float'
                || $arrayProps['type'] === 'int'
                || $arrayProps['type'] === 'numeric'
            ) {
                $result[] = ['', TypeError::class];
            } else {
                $result[] = ['', EmptyPropertyValueException::class];
            }
        }

        if (!isset($arrayProps['type'])) {
            return $result;
        }

        if ($arrayProps['type'] === Metadata::class) {
            if (isset($arrayProps['all_type']) && $arrayProps['all_type'] === 'string') {
                $arrayMetadata = [
                    [[Random::int()], InvalidPropertyValueTypeException::class],
                    [[new stdClass()], InvalidPropertyValueTypeException::class],
                    [true, ValidatorParameterException::class],
                ];
                return array_merge($result, $arrayMetadata);
            }
            $arrayMetadata = [
                [Random::int(), InvalidPropertyValueTypeException::class],
                [new stdClass(), InvalidPropertyValueTypeException::class],
                [true, InvalidPropertyValueTypeException::class],
            ];
            return array_merge($result, $arrayMetadata);
        }

        if ($arrayProps['type'] === 'DateTime') {
            $arrayDateTime = [
                [Random::str(100), InvalidPropertyValueException::class],
                [Random::int(), InvalidPropertyValueException::class],
            ];
            return array_merge($result, $arrayDateTime);
        }

        /**
         * object
         */
        if (class_exists($arrayProps['type']) && $arrayProps['type'] !== ListObject::class) {
            $arrayObject = [
                [new stdClass(), InvalidPropertyValueTypeException::class],
                [new DateTime(), InvalidPropertyValueTypeException::class],
            ];
            return array_merge($result, $arrayObject);
        }

        /**
         * ListObject
         */
        if (($arrayProps['type'] === ListObject::class) && isset($arrayProps['all_type'])) {
            return [
                [new stdClass(), ValidatorParameterException::class],
                [[new stdClass()], InvalidArgumentException::class],
            ];
        }

        /**
         * choices
         */
        if ($arrayProps['type'] === 'string' && isset($arrayProps['choices'])) {
            return [
                [[], TypeError::class],
                [fopen(__FILE__, 'rb'), TypeError::class],
                [true, InvalidPropertyValueException::class],
                [Random::str(10), InvalidPropertyValueException::class],
            ];
        }

        if ($arrayProps['type'] === 'bool') {
            return [
                [['string'], TypeError::class],
            ];
        }

        if ($arrayProps['type'] === 'string') {
            $arrayString = [
                [[], TypeError::class],
                [new stdClass(), TypeError::class],
            ];
            if (isset($arrayProps['max'])) {
                $result[] = [Random::str($arrayProps['max'] + 1), InvalidPropertyValueException::class];
            }
            if (isset($arrayProps['min']) && $arrayProps['min'] > 1) {
                $result[] = [Random::str($arrayProps['min'] - 1), InvalidPropertyValueException::class];
            }
            return array_merge($result, $arrayString);
        }

        if ($arrayProps['type'] === 'float' || $arrayProps['type'] === 'int' || $arrayProps['type'] === 'numeric') {
            $arrayFloat = [
                [[], TypeError::class],
                [new stdClass(), TypeError::class],
            ];
            if (isset($arrayProps['max'])) {
                $result[] = [Random::int($arrayProps['max'], $arrayProps['max'] + 1), InvalidPropertyValueException::class];
            }
            if (isset($arrayProps['min']) && $arrayProps['min'] > 1) {
                $result[] = [Random::int($arrayProps['min'] - 1, $arrayProps['min']), InvalidPropertyValueException::class];
            }
            if (isset($arrayProps['greaterThan'])) {
                $result[] = [$arrayProps['greaterThan'] - 1, InvalidPropertyValueException::class];
            }
            if (isset($arrayProps['lessThan'])) {
                $result[] = [$arrayProps['lessThan'] + 1, InvalidPropertyValueException::class];
            }
            return array_merge($result, $arrayFloat);
        }

        if ($arrayProps['type'] === 'array') {
            $array = [
                [[new stdClass()], InvalidPropertyValueTypeException::class],
            ];
            return array_merge($result, $array);
        }

        return $result;
    }

    /**
     * Парсит массив правил свойства
     *
     * @param array $propRules
     *
     * @return array
     */
    private function parsePropRules(array $propRules): array
    {
        $arrayProps = [
            'isRequired' => false
        ];
        foreach ($propRules as $rule) {
            switch (true) {
                case $rule instanceof Length:
                    $arrayProps['max'] = (int)$rule->getMax() ?: $arrayProps['max'] ?? null;
                    $arrayProps['min'] = (int)$rule->getMin() ?: $arrayProps['min'] ?? null;
                    break;
                case $rule instanceof NotBlank:
                case $rule instanceof NotNull:
                    $arrayProps['isRequired'] = true;
                    break;
                case $rule instanceof Choice:
                    $arrayProps['choices'] = $rule->getCallback() ?: $rule->getChoices();
                    break;
                case $rule instanceof GreaterThanOrEqual:
                    $arrayProps['greaterThanOrEqual'] = $rule->getValue();
                    break;
                case $rule instanceof GreaterThan:
                    $arrayProps['greaterThan'] = $rule->getValue();
                    break;
                case $rule instanceof LessThanOrEqual:
                    $arrayProps['lessThanOrEqual'] = $rule->getValue();
                    break;
                case $rule instanceof LessThan:
                    $arrayProps['lessThan'] = $rule->getValue();
                    break;
                case $rule instanceof Regex:
                    $arrayProps['regex'] = str_replace('/', '', $rule->getPattern());
                    break;
                case $rule instanceof Type:
                    $arrayProps['type'] = $rule->getType();
                    break;
                case $rule instanceof AllType:
                    $arrayProps['all_type'] = $rule->getType();
                    break;
                case $rule instanceof DateTimeValidation:
                    $arrayProps['datetime_format'] = $rule->getFormat();
                    break;
                case $rule instanceof Count:
                    $arrayProps['count_max'] = (int)$rule->getMax() ?: $arrayProps['count_max'] ?? null;
                    $arrayProps['count_min'] = (int)$rule->getMin() ?: $arrayProps['count_min'] ?? null;
                    break;
                case $rule instanceof Url:
                    $arrayProps['url'] = $rule->getProtocols();
                    break;
                case $rule instanceof Date:
                    $arrayProps['data'] = true;
                    break;
                case $rule instanceof Email:
                    $arrayProps['email'] = $rule->getMode();
                    break;
                case $rule instanceof Ip:
                    $arrayProps['ip'] = $rule->getVersion();
                    break;
            }
        }

        return $arrayProps;
    }

    /**
     * Подготавливает валидные данные на основе правил свойства
     *
     * @param array $arrayProps
     *
     * @return array
     * @throws \ReflectionException
     */
    private function prepareDataByType(array $arrayProps): array
    {
        if (!isset($arrayProps['type'])) {
            return [];
        }

        $result = [];

        /** Генерирует массив объектов */
        if (isset($arrayProps['all_type']) && class_exists($arrayProps['all_type'])) {
            return $this->generateArrayOfObject($arrayProps);
        }

        /** Генерирует массив массивов */
        if (class_exists($arrayProps['type']) && $arrayProps['type'] !== 'DateTime' && $arrayProps['type'] !== Metadata::class) {
            return $this->generateArrayOfArray($arrayProps);
        }

        $max = Random::int(1, 10);
        for ($i = 0; $i < $max; $i++) {
            $result[] = [$this->generateRandom($arrayProps)];
        }
        return $result;
    }

    /**
     * Подготавливает валидные данные на основе регулярного выражения
     *
     * @param string $regex
     * @param int|null $max
     * @param int|null $min
     * @return string|null
     */
    private function prepareDataByRegex(string $regex, ?int $max = null, ?int $min = null): ?string
    {
        $regex = preg_replace('/^\/?\^?/', '', $regex);
        $regex = preg_replace('/([\^\$])|[^ \]\{\}\)]+$/', '', $regex);
        if (($min || $max) && !preg_match('/{((.*?))}/', $regex)) {
            $length = '{' . ($min !== null ? $min : '') . ',' . ($max !== null ? $max : '') . '}';
            $regex .= $length;
        }

        return Factory::create()->regexify($regex);
    }

    /**
     * Подготавливает массив данных для объекта на основе рефлексии класса
     *
     * @param string|object $class Класс, данные для которых необходимо подготовить
     * @param bool $isArray Флаг создания данных в виде объекта или массива
     *
     * @return array
     * @throws Exception
     * @throws \ReflectionException
     */
    private function prepareArrayForObject(mixed $class, bool $isArray = true): array
    {
        $value = [];
        $object = new $class;
        $validator = new Validator($object);
        $properties = (new ReflectionClass($object))->getProperties();
        foreach ($properties as $property) {
            $arrayProps = $this->parsePropRules($validator->getRulesByPropName($property->getName()));
            if (!isset($arrayProps['type'])) {
                return [];
            }
            $propertyName = substr($property->getName(), 1);

            /** Проверяем, является ли параметр массивом объектов */
            $class = $arrayProps['all_type'] ?? $arrayProps['type'];

            /** Проверяем, установлено ли дефолтное значение в свойстве, если да, то устанавливаем и выходим */
            if (!empty($object->{$property->getName()}) && !class_exists($class) && ($class === 'string' || (int)$object->{$property->getName()} !== 0 )) {
                $value[$propertyName] = $object->{$property->getName()};
                continue;
            }

            /** Проверяем, является ли значение $class абстрактным классом, если да, то генерируем модель через фабрику, заполняем её данными и выходим */
            if (class_exists($class)) {
                $reflectionProperty = new ReflectionClass($class);
                if ($reflectionProperty->isAbstract()) {
                    $class = $this->generateClassByFactory($reflectionProperty);
                    $value[$propertyName] = $this->prepareArrayForObject($class);
                    continue;
                }
            }

            /** Проверяем, является ли параметр массивом объектов, если да, то генерируем массив массивов\объектов и выходим */
            if (isset($arrayProps['all_type']) && $arrayProps['all_type'] !== 'string') {
                $value[$propertyName] = $isArray ? [$this->prepareArrayForObject($class, $isArray)] : [new $class($this->prepareArrayForObject($class, $isArray))];
                continue;
            }

            /** Проверяем, является ли значение $class классом\DateTime\Metadata, если нет\да, то устанавливаем рандомные значения для свойства и выходим */
            if ($class === 'DateTime' || $class === Metadata::class || !class_exists($class)) {
                $value[$propertyName] = isset($arrayProps['regex']) ? $this->prepareDataByRegex($arrayProps['regex'], $arrayProps['max'] ?? null, isset($arrayProps['min']) ?? null) : $this->generateRandom($arrayProps);
                continue;
            }

            /** Проверяем, является ли значение $class enum, если да, то выбираем рандомно значение и выходим */
            if (new $class instanceof AbstractEnum) {
                /** @var AbstractEnum $class */
                $value[$propertyName] = Random::value($class::getValidValues());
                continue;
            }

            /** Проверяем, является ли значение $class классом, если да, то генерируем значения в виде массивов или объектов и выходим */
            if (new $class instanceof AbstractObject) {
                $value[$propertyName] = $this->prepareArrayForObject($class);
            }
        }

        return $value;
    }

    /**
     * Генерирует рандомные данные на основе типа свойства
     *
     * @param array $arrayProps
     *
     * @return mixed
     */
    private function generateRandom(array $arrayProps): mixed
    {
        if (!isset($arrayProps['type'])) {
            return null;
        }

        $faker = Factory::create();

        if ($arrayProps['type'] === 'string' && isset($arrayProps['choices'])) {
            if (class_exists($arrayProps['choices'][0])) {
                /** @var AbstractEnum $enumClass */
                $enumClass = new $arrayProps['choices'][0]();
                return $faker->randomElement($enumClass::getValidValues());
            }

            if (is_array($arrayProps['choices'])) {
                return $faker->randomElement($arrayProps['choices']);
            }
        }

        if ($arrayProps['type'] === 'string') {
            if (isset($arrayProps['url'])) {
                $max = isset($arrayProps['max']) ? $arrayProps['max'] - 11 : 10;
                $min = $arrayProps['min'] ?? 5;

                return 'https://' . Random::str($min, $max, 'abcdefghilkmnopqrstuvwxyz') . '.ru';
            }

            if (isset($arrayProps['email'])) {
                return $faker->email();
            }

            if (isset($arrayProps['ip'])) {
                return $arrayProps['ip'] === Ip::V6 ? $faker->ipv6() : $faker->ipv4();
            }

            return Random::str($arrayProps['min'] ?? 1, $arrayProps['max'] ?? 100);
        }

        if ($arrayProps['type'] === 'float' || $arrayProps['type'] === 'numeric') {
            $floatResult = $faker->randomFloat(null, $arrayProps['min'] ?? $arrayProps['greaterThanOrEqual'] ?? 1, $arrayProps['max'] ?? $arrayProps['lessThanOrEqual'] ?? 1000);
            return (int) round($floatResult * 100.0);
        }

        if ($arrayProps['type'] === 'int') {
            return $faker->numberBetween($arrayProps['min'] ?? $arrayProps['greaterThanOrEqual'] ?? 1, $arrayProps['max'] ?? $arrayProps['lessThanOrEqual'] ?? 1000);
        }

        if ($arrayProps['type'] === 'bool') {
            return $faker->boolean();
        }

        if ($arrayProps['type'] === 'DateTime') {
            $format = (isset($arrayProps['data']) && $arrayProps['data'])
                ? 'Y-m-d'
                : $arrayProps['datetime_format'] ?? YOOKASSA_DATE;
            return $faker->boolean() ? $faker->time($format) : $faker->date($format);
        }

        if ($arrayProps['type'] === 'array' || $arrayProps['type'] === Metadata::class) {
            $arrayData = [];
            $max = $faker->numberBetween(1, 10);
            for ($i = 0; $i < $max; $i++) {
                $arrayData[Random::str(1, 10)] = Random::str(5, 10);
            }
            return $arrayData;
        }

        if (class_exists($arrayProps['type'])) {
            return new $arrayProps['type'];
        }

        return null;
    }

    /**
     * Генерирует экземпляр класса на основе фабрики, беря рандомные данные из фабрики
     *
     * @param ReflectionClass $reflection
     *
     * @return mixed
     * @throws \ReflectionException
     */
    private function generateClassByFactory(ReflectionClass $reflection): mixed
    {
        $fileName = substr($reflection->getShortName(), 8);
        $factoryName = $reflection->getNamespaceName() . '\\' . $fileName . 'Factory';
        $factoryClass = new $factoryName();
        $reflectionFactoryClass = new ReflectionClass($factoryClass);
        $typeClassMap = $reflectionFactoryClass->getProperty('typeClassMap')->getDefaultValue();

        $typeValue = Random::value(array_keys($typeClassMap));
        return (new $factoryName)->factory($typeValue);
    }

    /**
     * Генерирует массив объектов
     *
     * @param array $arrayProps
     *
     * @return array
     * @throws \ReflectionException
     */
    private function generateArrayOfObject(array $arrayProps): array
    {
        $result = [];
        $value = [];
        $max = Random::int(1, 10);
        for ($y = 0; $y < $max; $y++) {
            /** Проверяем, есть ли требование по минимальному и максимальному количеству объектов в коллекции, если да, берем эти значения, если нет, берем рандомно */
            $countMax = Random::int($arrayProps['count_min'] ?? 1, $arrayProps['count_max'] ?? 10);
            for ($i = 0; $i < $countMax; $i++) {
                $isArray = Random::bool();
                $value[] = $i % 2 ? new $arrayProps['all_type']($this->prepareArrayForObject($arrayProps['all_type'], $isArray)) : $this->prepareArrayForObject($arrayProps['all_type']);
            }
            $result[] = [$value];
            unset($value);
        }
        return $result;
    }

    /**
     * Генерирует массив массивов
     *
     * @param array $arrayProps
     *
     * @return array
     * @throws \ReflectionException
     */
    private function generateArrayOfArray(array $arrayProps): array
    {
        $result = [];
        $reflection = new ReflectionClass($arrayProps['type']);
        $max = Random::int(1, 10);
        for ($i = 0; $i < $max; $i++) {
            $isArray = Random::bool();
            $arrayProps['type'] = !$reflection->isAbstract() ? $reflection->getName() : $this->generateClassByFactory($reflection);
            $result[] = [$i % 2 ? new $arrayProps['type']($this->prepareArrayForObject($arrayProps['type'], $isArray)) : $this->prepareArrayForObject($arrayProps['type'])];
        }
        return $result;
    }
}
