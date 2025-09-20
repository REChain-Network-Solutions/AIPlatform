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

namespace YooKassa\Request\Deals;

use DateTime;
use Exception;
use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;

/**
 * Класс, представляющий модель DealsRequestBuilder.
 *
 * Класс билдера запросов к API для получения списка сделок магазина.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class DealsRequestBuilder extends AbstractRequestBuilder
{
    /**
     * @var DealsRequest|null Собираемый объект запроса списка сделок магазина
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Устанавливает ограничение количества объектов сделки.
     *
     * @param null|string|int $value Ограничение количества объектов сделки или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setLimit(mixed $value): DealsRequestBuilder
    {
        $this->currentObject->setLimit($value);

        return $this;
    }

    /**
     * Устанавливает страница выдачи результатов.
     *
     * @param null|string $value Страница выдачи результатов или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setCursor(?string $value): DealsRequestBuilder
    {
        $this->currentObject->setCursor($value);

        return $this;
    }

    /**
     * Устанавливает дату создания от которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время создания, от (не включая) или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setCreatedAtGt($value): DealsRequestBuilder
    {
        $this->currentObject->setCreatedAtGt($value);

        return $this;
    }

    /**
     * Устанавливает дату создания от которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время создания, от (включительно) или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setCreatedAtGte($value): DealsRequestBuilder
    {
        $this->currentObject->setCreatedAtGte($value);

        return $this;
    }

    /**
     * Устанавливает дату создания до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время создания, до (не включая) или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setCreatedAtLt($value): DealsRequestBuilder
    {
        $this->currentObject->setCreatedAtLt($value);

        return $this;
    }

    /**
     * Устанавливает дату создания до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время создания, до (включительно) или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setCreatedAtLte($value): DealsRequestBuilder
    {
        $this->currentObject->setCreatedAtLte($value);

        return $this;
    }

    /**
     * Устанавливает дату автоматического закрытия от которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время автоматического закрытия, до (включительно) или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setExpiresAtGt($value): DealsRequestBuilder
    {
        $this->currentObject->setExpiresAtGt($value);

        return $this;
    }

    /**
     * Устанавливает дату автоматического закрытия от которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время автоматического закрытия, от (включительно) или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setExpiresAtGte($value): DealsRequestBuilder
    {
        $this->currentObject->setExpiresAtGte($value);

        return $this;
    }

    /**
     * Устанавливает дату автоматического закрытия до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время автоматического закрытия, до (включительно) или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setExpiresAtLt($value): DealsRequestBuilder
    {
        $this->currentObject->setExpiresAtLt($value);

        return $this;
    }

    /**
     * Устанавливает дату автоматического закрытия до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время автоматического закрытия, до (включительно) или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setExpiresAtLte($value): DealsRequestBuilder
    {
        $this->currentObject->setExpiresAtLte($value);

        return $this;
    }

    /**
     * Устанавливает статус выбираемых сделок.
     *
     * @param string|null $value Статус выбираемых сделок или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setStatus(?string $value): DealsRequestBuilder
    {
        $this->currentObject->setStatus($value);

        return $this;
    }

    /**
     * Устанавливает фильтр по описанию выбираемых сделок.
     *
     * @param string|null $value Фильтр по описанию выбираемых сделок или null, чтобы удалить значение
     *
     * @return DealsRequestBuilder Инстанс текущего билдера
     */
    public function setFullTextSearch(?string $value): DealsRequestBuilder
    {
        $this->currentObject->setFullTextSearch($value);

        return $this;
    }

    /**
     * Собирает и возвращает объект запроса списка сделок магазина.
     *
     * @param null|array $options Массив с настройками запроса
     *
     * @return AbstractRequestInterface Инстанс объекта запроса к API для получения списка сделок магазина
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        return parent::build($options);
    }

    /**
     * Возвращает новый объект запроса для получения списка сделок, который в дальнейшем будет собираться в билдере.
     *
     * @return DealsRequest Объект запроса списка сделок магазина
     */
    protected function initCurrentObject(): DealsRequest
    {
        return new DealsRequest();
    }
}
