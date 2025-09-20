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

namespace YooKassa\Common;

use ArrayAccess;
use DateTime;
use JsonSerializable;
use ReturnTypeWillChange;
use Traversable;
use YooKassa\Helpers\TypeCast;
use YooKassa\Validator\Constraints as Assert;
use YooKassa\Validator\Validator;

if (!defined('YOOKASSA_DATE')) {
    define('YOOKASSA_DATE', 'Y-m-d\\TH:i:s.v\\Z');
}

/**
 * Класс, представляющий модель AbstractObject.
 *
 * Базовый класс генерируемых объектов.
 *
 * @category Class
 * @package  YooKassa
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
abstract class AbstractObject implements ArrayAccess, JsonSerializable
{
    /**
     * @var Validator Валидатор данных
     */
    private Validator $validator;

    /**
     * @var array Свойства установленные пользователем
     */
    private array $unknownProperties = [];

    /**
     * AbstractObject constructor.
     * @param array|null $data
     */
    public function __construct(?array $data = [])
    {
        $this->validator = new Validator($this);
        if (!empty($data) && is_array($data)) {
            $this->fromArray($data);
        }
    }

    /**
     * Возвращает значение свойства.
     *
     * @param string $propertyName Имя свойства
     *
     * @return mixed Значение свойства
     */
    public function __get(string $propertyName): mixed
    {
        return $this->offsetGet($propertyName);
    }

    /**
     * Устанавливает значение свойства.
     *
     * @param string $propertyName Имя свойства
     * @param mixed $value Значение свойства
     */
    public function __set(string $propertyName, mixed $value): void
    {
        $this->offsetSet($propertyName, $value);
    }

    /**
     * Проверяет наличие свойства.
     *
     * @param string $propertyName Имя проверяемого свойства
     *
     * @return bool True если свойство имеется, false если нет
     */
    public function __isset(string $propertyName): bool
    {
        return $this->offsetExists($propertyName);
    }

    /**
     * Удаляет свойство.
     *
     * @param string $propertyName Имя удаляемого свойства
     */
    public function __unset(string $propertyName): void
    {
        $this->offsetUnset($propertyName);
    }

    #[ReturnTypeWillChange]
    /**
     * Проверяет наличие свойства.
     *
     * @param string $offset Имя проверяемого свойства
     *
     * @return bool True если свойство имеется, false если нет
     */
    public function offsetExists(mixed $offset): bool
    {
        $method = 'get' . ucfirst($offset);
        if (method_exists($this, $method)) {
            return true;
        }
        $method = 'get' . self::matchPropertyName($offset);
        if (method_exists($this, $method)) {
            return true;
        }

        return array_key_exists($offset, $this->unknownProperties);
    }

    #[ReturnTypeWillChange]
    /**
     * Возвращает значение свойства.
     *
     * @param string $offset Имя свойства
     *
     * @return mixed Значение свойства
     */
    public function offsetGet(mixed $offset): mixed
    {
        if ($offset === 'validator') {
            return null;
        }
        $method = 'get' . ucfirst($offset);
        if (method_exists($this, $method)) {
            return $this->{$method}();
        }
        $method = 'get' . self::matchPropertyName($offset);
        if (method_exists($this, $method)) {
            return $this->{$method}();
        }

        return $this->unknownProperties[$offset] ?? null;
    }

    #[ReturnTypeWillChange]
    /**
     * Устанавливает значение свойства.
     *
     * @param string $offset Имя свойства
     * @param mixed $value Значение свойства
     */
    public function offsetSet(mixed $offset, mixed $value): void
    {
        if ($offset === 'validator') {
            return;
        }
        $method = 'set' . ucfirst($offset);
        if (method_exists($this, $method)) {
            $this->{$method}($value);
        } else {
            $method = 'set' . self::matchPropertyName($offset);
            if (method_exists($this, $method)) {
                $this->{$method}($value);
            } else {
                $this->unknownProperties[$offset] = $value;
            }
        }
    }

    #[ReturnTypeWillChange]
    /**
     * Удаляет свойство.
     *
     * @param string $offset Имя удаляемого свойства
     */
    public function offsetUnset(mixed $offset): void
    {
        if ($offset === 'validator') {
            return;
        }
        $method = 'set' . ucfirst($offset);
        if (method_exists($this, $method)) {
            $this->{$method}(null);
        } else {
            $method = 'set' . self::matchPropertyName($offset);
            if (method_exists($this, $method)) {
                $this->{$method}(null);
            } else {
                unset($this->unknownProperties[$offset]);
            }
        }
    }

    /**
     * Устанавливает значения свойств текущего объекта из массива.
     *
     * @param array|Traversable $sourceArray Ассоциативный массив с настройками
     */
    public function fromArray(iterable $sourceArray): void
    {
        foreach ($sourceArray as $key => $value) {
            $this->offsetSet($key, $value);
        }
    }

    /**
     * Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
     * Является алиасом метода AbstractObject::jsonSerialize().
     *
     * @return array Ассоциативный массив со свойствами текущего объекта
     */
    public function toArray(): array
    {
        return $this->jsonSerialize();
    }

    #[ReturnTypeWillChange]
    /**
     * Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.
     *
     * @return array Ассоциативный массив со свойствами текущего объекта
     */
    public function jsonSerialize(): array
    {
        $excludedMethods = ['getUnknownProperties', 'getIterator', 'getValidator'];
        $result = [];
        foreach (get_class_methods($this) as $method) {
            if (0 === strncmp('get', $method, 3)) {
                if (in_array($method, $excludedMethods)) {
                    continue;
                }
                $property = strtolower(preg_replace('/[A-Z]/', '_\0', lcfirst(substr($method, 3))));
                $value = $this->serializeValueToJson($this->{$method}());
                if (null !== $value) {
                    $result[$property] = $value;
                }
            }
        }
        if (!empty($this->unknownProperties)) {
            foreach ($this->unknownProperties as $property => $value) {
                if (!array_key_exists($property, $result)) {
                    $result[$property] = $this->serializeValueToJson($value);
                }
            }
        }

        return $result;
    }

    /**
     * Возвращает массив свойств которые не существуют, но были заданы у объекта.
     *
     * @return array Ассоциативный массив с не существующими у текущего объекта свойствами
     */
    protected function getUnknownProperties(): array
    {
        return $this->unknownProperties;
    }

    private function serializeValueToJson($value)
    {
        if (null === $value || is_scalar($value)) {
            return $value;
        }
        if (is_array($value)) {
            $array = [];
            foreach ($value as $key => $item) {
                if ('validator' === $key) {
                    continue;
                }
                $array[$key] = $this->serializeValueToJson($item);
            }

            return $array;
        }
        if ($value instanceof JsonSerializable) {
            if ($value instanceof ListObjectInterface && $value->isEmpty()) {
                return null;
            }
            return $value->jsonSerialize();
        }
        if ($value instanceof DateTime) {
            return $value->format(YOOKASSA_DATE);
        }

        return $value;
    }

    /**
     * Преобразует имя свойства из snake_case в camelCase.
     *
     * @param string $property Преобразуемое значение
     *
     * @return string Значение в камэл кейсе
     */
    private static function matchPropertyName(string $property): string
    {
        return preg_replace('/_(\w)/', '\1', $property);
    }

    /**
     * @return Validator
     */
    public function getValidator(): Validator
    {
        return $this->validator;
    }

    /**
     * @param string $propertyName
     * @param mixed $propertyValue
     * @return mixed
     */
    protected function validatePropertyValue(string $propertyName, mixed $propertyValue): mixed
    {
        $propertyValue = $this->prepareTypeValue($propertyName, $propertyValue);
        $this->getValidator()->validatePropertyValue($propertyName, $propertyValue);
        return $propertyValue;
    }

    /**
     * @param string $propertyName
     * @param mixed $propertyValue
     * @return mixed
     */
    private function prepareTypeValue(string $propertyName, mixed $propertyValue): mixed
    {
        $finalType = null;
        $allType = null;
        foreach ($this->getValidator()->getRulesByPropName($propertyName) ?? [] as $rule) {
            switch (true) {
                case $rule instanceof Assert\Type:
                    $finalType = $rule->getType();
                    break;
                case $rule instanceof Assert\AllType:
                    $allType = $rule->getType();
                    break;
            }
        }
        if ($finalType === 'DateTime' && is_string($propertyValue) && TypeCast::canCastToDateTime($propertyValue)) {
            $this->getValidator()->validatePropertyValue($propertyName, $propertyValue, [Assert\Type::class]);
            return TypeCast::castToDateTime($propertyValue) ?: $propertyValue;
        }
        if ((is_string($propertyValue) || is_array($propertyValue)) && empty($propertyValue) && $propertyValue !== '0') {
            return null;
        }
        if (!is_null($allType) && is_array($propertyValue) && $finalType === ListObject::class && class_exists($allType)) {
            return new ListObject($allType, $propertyValue);
        }
        if (is_array($propertyValue) && class_exists($finalType) && new $finalType instanceof self) {
            return new $finalType($propertyValue);
        }

        return $propertyValue;
    }
}
