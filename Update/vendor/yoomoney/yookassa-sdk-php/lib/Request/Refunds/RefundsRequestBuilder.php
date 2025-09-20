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

namespace YooKassa\Request\Refunds;

use DateTime;
use Exception;
use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;

/**
 * Класс, представляющий модель RefundsRequestBuilder.
 *
 * Класс билдера объектов запросов к API списка возвратов.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class RefundsRequestBuilder extends AbstractRequestBuilder
{
    /**
     * @var RefundsRequest|null Инстанс собираемого объекта запроса
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Устанавливает идентификатор платежа или null, если требуется его удалить.
     *
     * @param null|string $value Идентификатор платежа
     *
     * @return RefundsRequestBuilder Инстанс текущего объекта билдера
     *
     * @throws InvalidPropertyValueException Выбрасывается если длина переданной строки не равна 36 символам
     * @throws InvalidPropertyValueTypeException Выбрасывается если в метод была передана не строка
     */
    public function setPaymentId(?string $value): RefundsRequestBuilder
    {
        $this->currentObject->setPaymentId($value);

        return $this;
    }

    /**
     * Устанавливает статус выбираемых возвратов.
     *
     * @param string $value Статус выбираемых платежей или null, чтобы удалить значение
     *
     * @return RefundsRequestBuilder Инстанс текущего объекта билдера
     *
     * @throws InvalidPropertyValueException Выбрасывается если переданное значение не является валидным статусом
     * @throws InvalidPropertyValueTypeException Выбрасывается если в метод была передана не строка
     */
    public function setStatus(string $value): RefundsRequestBuilder
    {
        $this->currentObject->setStatus($value);

        return $this;
    }

    /**
     * Устанавливает ограничение количества объектов возвратов.
     *
     * @param null|int $value Ограничение количества объектов возвратов или null, чтобы удалить значение
     *
     * @return RefundsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если в метод было передана не целое число
     */
    public function setLimit(?int $value): RefundsRequestBuilder
    {
        $this->currentObject->setLimit($value);

        return $this;
    }

    /**
     * Устанавливает токен следующей страницы выборки.
     *
     * @param string|null $value Токен следующей страницы выборки или null, чтобы удалить значение
     *
     * @return RefundsRequestBuilder Инстанс текущего объекта билдера
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если в метод была передана не строка
     */
    public function setCursor(?string $value): RefundsRequestBuilder
    {
        $this->currentObject->setCursor($value);

        return $this;
    }

    /**
     * Устанавливает дату создания от которой выбираются возвраты.
     *
     * @param null|DateTime|int|string $value Время создания, от (не включая) или null, чтобы удалить значение
     *
     * @return RefundsRequestBuilder Инстанс текущего объекта билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не строка, не число и не значение типа DateTime)
     */
    public function setCreatedAtGt(mixed $value): RefundsRequestBuilder
    {
        $this->currentObject->setCreatedAtGt($value);

        return $this;
    }

    /**
     * Устанавливает дату создания от которой выбираются возвраты.
     *
     * @param null|DateTime|int|string $value Время создания, от (включительно) или null, чтобы удалить значение
     *
     * @return RefundsRequestBuilder Инстанс текущего объекта билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана
     *                                       строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не
     *                                                     строка, не число и не значение типа DateTime)
     */
    public function setCreatedAtGte(mixed $value): RefundsRequestBuilder
    {
        $this->currentObject->setCreatedAtGte($value);

        return $this;
    }

    /**
     * Устанавливает дату создания до которой выбираются возвраты.
     *
     * @param null|DateTime|int|string $value Время создания, до (не включая) или null, чтобы удалить значение
     *
     * @return RefundsRequestBuilder Инстанс текущего объекта билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана
     *                                       строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не
     *                                                     строка, не число и не значение типа DateTime)
     */
    public function setCreatedAtLt(mixed $value): RefundsRequestBuilder
    {
        $this->currentObject->setCreatedAtLt($value);

        return $this;
    }

    /**
     * Устанавливает дату создания до которой выбираются возвраты.
     *
     * @param null|DateTime|int|string $value Время создания, до (включительно) или null, чтобы удалить значение
     *
     * @return RefundsRequestBuilder Инстанс текущего объекта билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана
     *                                       строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не
     *                                                     строка, не число и не значение типа DateTime)
     */
    public function setCreatedAtLte(mixed $value): RefundsRequestBuilder
    {
        $this->currentObject->setCreatedAtLte($value);

        return $this;
    }

    /**
     * Собирает и возвращает объект запроса списка возвратов магазина.
     *
     * @param null|array $options Массив с настройками запроса
     *
     * @return RefundsRequest Инстанс объекта запроса к API для получения списка возвратов магазина
     */
    public function build(?array $options = null): RefundsRequest
    {
        return parent::build($options);
    }

    /**
     * Инициализирует новый инстанс собираемого объекта.
     *
     * @return RefundsRequestInterface Инстанс собираемого запроса
     */
    protected function initCurrentObject(): AbstractRequestInterface
    {
        return new RefundsRequest();
    }
}
