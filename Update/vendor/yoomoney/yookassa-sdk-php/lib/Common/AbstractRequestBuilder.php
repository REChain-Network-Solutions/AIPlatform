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

use Exception;
use InvalidArgumentException;
use Traversable;
use YooKassa\Common\Exceptions\InvalidPropertyException;
use YooKassa\Common\Exceptions\InvalidRequestException;

/**
 * Класс, представляющий модель AbstractRequestBuilder.
 *
 * Базовый класс билдера запросов.
 *
 * @category Class
 * @package  YooKassa
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
abstract class AbstractRequestBuilder
{
    /**
     * Инстанс собираемого запроса.
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Конструктор, инициализирует пустой запрос, который в будущем начнём собирать.
     */
    public function __construct()
    {
        $this->currentObject = $this->initCurrentObject();
    }

    /**
     * Строит запрос, валидирует его и возвращает, если все прошло нормально.
     *
     * @param null|array $options Массив свойств запроса, если нужно их установить перед сборкой
     *
     * @return AbstractRequestInterface Инстанс собранного запроса
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        if (!empty($options)) {
            $this->setOptions($options);
        }

        try {
            $this->currentObject->clearValidationError();
            if (!$this->currentObject->validate()) {
                throw new InvalidRequestException($this->currentObject);
            }
        } catch (InvalidRequestException $e) {
            throw $e;
        } catch (Exception $e) {
            throw new InvalidRequestException($this->currentObject, 0, $e);
        }
        $result = $this->currentObject;
        $this->currentObject = $this->initCurrentObject();

        return $result;
    }

    /**
     * Устанавливает свойства запроса из массива.
     *
     * @param iterable|null $options Массив свойств запроса
     *
     * @return AbstractRequestBuilder Инстанс текущего билдера запросов
     *
     * @throws InvalidArgumentException Выбрасывается если аргумент не массив и не итерируемый объект
     * @throws InvalidPropertyException Выбрасывается если не удалось установить один из параметров, переданных в массиве настроек
     */
    public function setOptions(mixed $options): AbstractRequestBuilder
    {
        if (empty($options)) {
            return $this;
        }
        if (!is_array($options) && !($options instanceof Traversable)) {
            throw new InvalidArgumentException('Invalid options value in setOptions method');
        }
        foreach ($options as $property => $value) {
            $method = 'set' . ucfirst($property);
            if (method_exists($this, $method)) {
                $this->{$method}($value);
            } else {
                $property = str_replace('.', '_', $property);
                $field = implode('', array_map('ucfirst', explode('_', $property)));
                $method = 'set' . ucfirst($field);
                if (method_exists($this, $method)) {
                    $this->{$method}($value);
                }
            }
        }

        return $this;
    }

    /**
     * Инициализирует пустой запрос
     *
     * @return AbstractRequestInterface|null Инстанс запроса, который будем собирать
     */
    abstract protected function initCurrentObject(): ?AbstractRequestInterface;
}
