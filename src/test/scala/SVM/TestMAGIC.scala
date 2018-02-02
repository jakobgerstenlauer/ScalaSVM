package SVM

import scala.collection.mutable.ListBuffer
import scala.concurrent.{Await, Future}

object TestMAGIC extends App {

  val d = new LocalData()
  println(d)

  val workingDir = "/home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/"
  val pathTrain = workingDir + "magic04train.csv"
  val pathValidation = workingDir + "magic04validation.csv"
  val pathTest = workingDir + "magic04test.csv"

  //The labels are in the second column (the column index is 0 based)
  val columnIndexLabel = 11
  //The first column has to be skipped, it contains a line nr!!!
  val columnIndexLineNr = 0
  val transformLabel = (x:Double) => if(x<=0) -1 else +1
  d.readTrainingDataSet (pathTrain, ',', columnIndexLabel, transformLabel, columnIndexLineNr)
  d.readTestDataSet (pathTest, ',', columnIndexLabel, transformLabel, columnIndexLineNr)
  d.readValidationDataSet(pathValidation, ',', columnIndexLabel, transformLabel, columnIndexLineNr)
  d.readTestDataSet (pathTest, ',', columnIndexLabel, transformLabel, columnIndexLineNr)
  d.readValidationDataSet(pathValidation, ',', columnIndexLabel, transformLabel, columnIndexLineNr)
  d.tableLabels()

  val epsilon = 0.0001
  val kernelPar = GaussianKernelParameter(1.0)
  val gaussianKernel = GaussianKernel(kernelPar)
  val kmf = LeanMatrixFactory(d, gaussianKernel, epsilon)
  val mp = ModelParams(C = 100, delta = 0.01)
  val alphas = new Alphas(N=d.N_train, mp)
  val ap = AlgoParams(batchProb = 0.99, learningRateDecline = 0.5,
    epsilon = epsilon)
  var algo = NoMatrices(alphas, ap, mp, kmf, new ListBuffer[Future[(Int,Int,Int)]])
  var numInt = 0
  while(numInt < ap.maxIter){
    algo = algo.iterate(numInt)
    numInt += 1
  }
  val testSetAccuracy : Future[Int] = algo.predictOnTestSet(PredictionMethod.AUC)
  Await.result(testSetAccuracy, LeanMatrixFactory.maxDuration)
}
/*
/usr/lib/jvm/java-8-oracle/bin/java -Xmx14G -Xms14G -Djava.lang.Integer.IntegerCache.high=1000000 -javaagent:/home/jakob/idea-IC-173.3727.127/lib/idea_rt.jar=38654:/home/jakob/idea-IC-173.3727.127/bin -Dfile.encoding=UTF-8 -classpath /usr/lib/jvm/java-8-oracle/jre/lib/charsets.jar:/usr/lib/jvm/java-8-oracle/jre/lib/deploy.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/cldrdata.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/dnsns.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/jaccess.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/jfxrt.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/localedata.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/nashorn.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/sunec.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/sunjce_provider.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/sunpkcs11.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/zipfs.jar:/usr/lib/jvm/java-8-oracle/jre/lib/javaws.jar:/usr/lib/jvm/java-8-oracle/jre/lib/jce.jar:/usr/lib/jvm/java-8-oracle/jre/lib/jfr.jar:/usr/lib/jvm/java-8-oracle/jre/lib/jfxswt.jar:/usr/lib/jvm/java-8-oracle/jre/lib/jsse.jar:/usr/lib/jvm/java-8-oracle/jre/lib/management-agent.jar:/usr/lib/jvm/java-8-oracle/jre/lib/plugin.jar:/usr/lib/jvm/java-8-oracle/jre/lib/resources.jar:/usr/lib/jvm/java-8-oracle/jre/lib/rt.jar:/home/jakob/IdeaProjects/Dist_Online_Svm/target/scala-2.11/test-classes:/home/jakob/IdeaProjects/Dist_Online_Svm/target/scala-2.11/classes:/home/jakob/.ivy2/cache/aopalliance/aopalliance/jars/aopalliance-1.0.jar:/home/jakob/.ivy2/cache/org.scalatest/scalatest_2.11/bundles/scalatest_2.11-3.0.4.jar:/home/jakob/.ivy2/cache/org.scalactic/scalactic_2.11/bundles/scalactic_2.11-3.0.4.jar:/home/jakob/.ivy2/cache/org.scala-lang.modules/scala-xml_2.11/bundles/scala-xml_2.11-1.0.5.jar:/home/jakob/.ivy2/cache/bouncycastle/bcmail-jdk14/jars/bcmail-jdk14-138.jar:/home/jakob/.ivy2/cache/bouncycastle/bcprov-jdk14/jars/bcprov-jdk14-138.jar:/home/jakob/.ivy2/cache/com.chuusai/shapeless_2.11/bundles/shapeless_2.11-2.3.2.jar:/home/jakob/.ivy2/cache/com.clearspring.analytics/stream/jars/stream-2.7.0.jar:/home/jakob/.ivy2/cache/com.esotericsoftware/kryo-shaded/bundles/kryo-shaded-3.0.3.jar:/home/jakob/.ivy2/cache/com.esotericsoftware/minlog/bundles/minlog-1.3.0.jar:/home/jakob/.ivy2/cache/com.fasterxml.jackson.core/jackson-annotations/bundles/jackson-annotations-2.6.5.jar:/home/jakob/.ivy2/cache/com.fasterxml.jackson.core/jackson-core/bundles/jackson-core-2.6.5.jar:/home/jakob/.ivy2/cache/com.fasterxml.jackson.core/jackson-databind/bundles/jackson-databind-2.6.5.jar:/home/jakob/.ivy2/cache/com.fasterxml.jackson.module/jackson-module-paranamer/bundles/jackson-module-paranamer-2.6.5.jar:/home/jakob/.ivy2/cache/com.fasterxml.jackson.module/jackson-module-scala_2.11/bundles/jackson-module-scala_2.11-2.6.5.jar:/home/jakob/.ivy2/cache/com.github.fommil/jniloader/jars/jniloader-1.1.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/core/jars/core-1.1.2.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/native_ref-java/jars/native_ref-java-1.1.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/native_system-java/jars/native_system-java-1.1.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-linux-armhf/jars/netlib-native_ref-linux-armhf-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-linux-i686/jars/netlib-native_ref-linux-i686-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-linux-x86_64/jars/netlib-native_ref-linux-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-osx-x86_64/jars/netlib-native_ref-osx-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-win-i686/jars/netlib-native_ref-win-i686-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-win-x86_64/jars/netlib-native_ref-win-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-linux-armhf/jars/netlib-native_system-linux-armhf-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-linux-i686/jars/netlib-native_system-linux-i686-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-linux-x86_64/jars/netlib-native_system-linux-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-osx-x86_64/jars/netlib-native_system-osx-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-win-i686/jars/netlib-native_system-win-i686-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-win-x86_64/jars/netlib-native_system-win-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.rwl/jtransforms/jars/jtransforms-2.4.0.jar:/home/jakob/.ivy2/cache/com.google.code.findbugs/jsr305/jars/jsr305-1.3.9.jar:/home/jakob/.ivy2/cache/com.google.code.gson/gson/jars/gson-2.2.4.jar:/home/jakob/.ivy2/cache/com.google.guava/guava/jars/guava-11.0.2.jar:/home/jakob/.ivy2/cache/com.google.inject/guice/jars/guice-3.0.jar:/home/jakob/.ivy2/cache/com.google.protobuf/protobuf-java/bundles/protobuf-java-2.5.0.jar:/home/jakob/.ivy2/cache/com.jamesmurty.utils/java-xmlbuilder/jars/java-xmlbuilder-1.0.jar:/home/jakob/.ivy2/cache/com.lowagie/itext/jars/itext-2.1.5.jar:/home/jakob/.ivy2/cache/com.ning/compress-lzf/bundles/compress-lzf-1.0.3.jar:/home/jakob/.ivy2/cache/com.thoughtworks.paranamer/paranamer/jars/paranamer-2.6.jar:/home/jakob/.ivy2/cache/com.twitter/chill-java/jars/chill-java-0.8.0.jar:/home/jakob/.ivy2/cache/com.twitter/chill_2.11/jars/chill_2.11-0.8.0.jar:/home/jakob/.ivy2/cache/com.univocity/univocity-parsers/jars/univocity-parsers-2.2.1.jar:/home/jakob/.ivy2/cache/commons-beanutils/commons-beanutils/jars/commons-beanutils-1.7.0.jar:/home/jakob/.ivy2/cache/commons-beanutils/commons-beanutils-core/jars/commons-beanutils-core-1.8.0.jar:/home/jakob/.ivy2/cache/commons-cli/commons-cli/jars/commons-cli-1.2.jar:/home/jakob/.ivy2/cache/commons-codec/commons-codec/jars/commons-codec-1.10.jar:/home/jakob/.ivy2/cache/commons-collections/commons-collections/jars/commons-collections-3.2.2.jar:/home/jakob/.ivy2/cache/commons-configuration/commons-configuration/jars/commons-configuration-1.6.jar:/home/jakob/.ivy2/cache/commons-digester/commons-digester/jars/commons-digester-1.8.jar:/home/jakob/.ivy2/cache/commons-httpclient/commons-httpclient/jars/commons-httpclient-3.1.jar:/home/jakob/.ivy2/cache/commons-io/commons-io/jars/commons-io-2.4.jar:/home/jakob/.ivy2/cache/commons-lang/commons-lang/jars/commons-lang-2.6.jar:/home/jakob/.ivy2/cache/commons-logging/commons-logging/jars/commons-logging-1.0.4.jar:/home/jakob/.ivy2/cache/commons-net/commons-net/jars/commons-net-2.2.jar:/home/jakob/.ivy2/cache/io.dropwizard.metrics/metrics-core/bundles/metrics-core-3.1.2.jar:/home/jakob/.ivy2/cache/io.dropwizard.metrics/metrics-graphite/bundles/metrics-graphite-3.1.2.jar:/home/jakob/.ivy2/cache/io.dropwizard.metrics/metrics-json/bundles/metrics-json-3.1.2.jar:/home/jakob/.ivy2/cache/io.dropwizard.metrics/metrics-jvm/bundles/metrics-jvm-3.1.2.jar:/home/jakob/.ivy2/cache/io.netty/netty/bundles/netty-3.9.9.Final.jar:/home/jakob/.ivy2/cache/io.netty/netty-all/jars/netty-all-4.0.43.Final.jar:/home/jakob/.ivy2/cache/javax.activation/activation/jars/activation-1.1.1.jar:/home/jakob/.ivy2/cache/javax.annotation/javax.annotation-api/jars/javax.annotation-api-1.2.jar:/home/jakob/.ivy2/cache/javax.inject/javax.inject/jars/javax.inject-1.jar:/home/jakob/.ivy2/cache/javax.mail/mail/jars/mail-1.4.7.jar:/home/jakob/.ivy2/cache/javax.servlet/javax.servlet-api/jars/javax.servlet-api-3.1.0.jar:/home/jakob/.ivy2/cache/javax.validation/validation-api/jars/validation-api-1.1.0.Final.jar:/home/jakob/.ivy2/cache/javax.ws.rs/javax.ws.rs-api/jars/javax.ws.rs-api-2.0.1.jar:/home/jakob/.ivy2/cache/javax.xml.bind/jaxb-api/jars/jaxb-api-2.2.2.jar:/home/jakob/.ivy2/cache/javax.xml.stream/stax-api/jars/stax-api-1.0-2.jar:/home/jakob/.ivy2/cache/jfree/jcommon/jars/jcommon-1.0.16.jar:/home/jakob/.ivy2/cache/jfree/jfreechart/jars/jfreechart-1.0.13.jar:/home/jakob/.ivy2/cache/jline/jline/jars/jline-0.9.94.jar:/home/jakob/.ivy2/cache/log4j/log4j/bundles/log4j-1.2.17.jar:/home/jakob/.ivy2/cache/mx4j/mx4j/jars/mx4j-3.0.2.jar:/home/jakob/.ivy2/cache/net.iharder/base64/jars/base64-2.3.8.jar:/home/jakob/.ivy2/cache/net.java.dev.jets3t/jets3t/jars/jets3t-0.9.3.jar:/home/jakob/.ivy2/cache/net.jpountz.lz4/lz4/jars/lz4-1.3.0.jar:/home/jakob/.ivy2/cache/net.razorvine/pyrolite/jars/pyrolite-4.13.jar:/home/jakob/.ivy2/cache/net.sf.opencsv/opencsv/jars/opencsv-2.3.jar:/home/jakob/.ivy2/cache/net.sf.py4j/py4j/jars/py4j-0.10.4.jar:/home/jakob/.ivy2/cache/net.sourceforge.f2j/arpack_combined_all/jars/arpack_combined_all-0.1-javadoc.jar:/home/jakob/.ivy2/cache/net.sourceforge.f2j/arpack_combined_all/jars/arpack_combined_all-0.1.jar:/home/jakob/.ivy2/cache/org.antlr/antlr4-runtime/jars/antlr4-runtime-4.5.3.jar:/home/jakob/.ivy2/cache/org.apache.avro/avro/jars/avro-1.7.7.jar:/home/jakob/.ivy2/cache/org.apache.avro/avro-ipc/jars/avro-ipc-1.7.7.jar:/home/jakob/.ivy2/cache/org.apache.avro/avro-ipc/jars/avro-ipc-1.7.7-tests.jar:/home/jakob/.ivy2/cache/org.apache.avro/avro-mapred/jars/avro-mapred-1.7.7-hadoop2.jar:/home/jakob/.ivy2/cache/org.apache.commons/commons-compress/jars/commons-compress-1.4.1.jar:/home/jakob/.ivy2/cache/org.apache.commons/commons-crypto/jars/commons-crypto-1.0.0.jar:/home/jakob/.ivy2/cache/org.apache.commons/commons-lang3/jars/commons-lang3-3.5.jar:/home/jakob/.ivy2/cache/org.apache.commons/commons-math3/jars/commons-math3-3.4.1.jar:/home/jakob/.ivy2/cache/org.apache.curator/curator-client/bundles/curator-client-2.6.0.jar:/home/jakob/.ivy2/cache/org.apache.curator/curator-framework/bundles/curator-framework-2.6.0.jar:/home/jakob/.ivy2/cache/org.apache.curator/curator-recipes/bundles/curator-recipes-2.6.0.jar:/home/jakob/.ivy2/cache/org.apache.directory.api/api-asn1-api/bundles/api-asn1-api-1.0.0-M20.jar:/home/jakob/.ivy2/cache/org.apache.directory.api/api-util/bundles/api-util-1.0.0-M20.jar:/home/jakob/.ivy2/cache/org.apache.directory.server/apacheds-i18n/bundles/apacheds-i18n-2.0.0-M15.jar:/home/jakob/.ivy2/cache/org.apache.directory.server/apacheds-kerberos-codec/bundles/apacheds-kerberos-codec-2.0.0-M15.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-annotations/jars/hadoop-annotations-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-auth/jars/hadoop-auth-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-client/jars/hadoop-client-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-common/jars/hadoop-common-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-hdfs/jars/hadoop-hdfs-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-mapreduce-client-app/jars/hadoop-mapreduce-client-app-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-mapreduce-client-common/jars/hadoop-mapreduce-client-common-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-mapreduce-client-core/jars/hadoop-mapreduce-client-core-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-mapreduce-client-jobclient/jars/hadoop-mapreduce-client-jobclient-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-mapreduce-client-shuffle/jars/hadoop-mapreduce-client-shuffle-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-yarn-api/jars/hadoop-yarn-api-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-yarn-client/jars/hadoop-yarn-client-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-yarn-common/jars/hadoop-yarn-common-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-yarn-server-common/jars/hadoop-yarn-server-common-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.httpcomponents/httpclient/jars/httpclient-4.3.6.jar:/home/jakob/.ivy2/cache/org.apache.httpcomponents/httpcore/jars/httpcore-4.3.3.jar:/home/jakob/.ivy2/cache/org.apache.ivy/ivy/jars/ivy-2.4.0.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-column/jars/parquet-column-1.8.2.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-common/jars/parquet-common-1.8.2.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-encoding/jars/parquet-encoding-1.8.2.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-format/jars/parquet-format-2.3.1.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-hadoop/jars/parquet-hadoop-1.8.2.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-jackson/jars/parquet-jackson-1.8.2.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-catalyst_2.11/jars/spark-catalyst_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-core_2.11/jars/spark-core_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-graphx_2.11/jars/spark-graphx_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-launcher_2.11/jars/spark-launcher_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-mllib-local_2.11/jars/spark-mllib-local_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-mllib_2.11/jars/spark-mllib_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-network-common_2.11/jars/spark-network-common_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-network-shuffle_2.11/jars/spark-network-shuffle_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-sketch_2.11/jars/spark-sketch_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-sql_2.11/jars/spark-sql_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-streaming_2.11/jars/spark-streaming_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-tags_2.11/jars/spark-tags_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-unsafe_2.11/jars/spark-unsafe_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.xbean/xbean-asm5-shaded/bundles/xbean-asm5-shaded-4.4.jar:/home/jakob/.ivy2/cache/org.apache.xmlgraphics/xmlgraphics-commons/jars/xmlgraphics-commons-1.3.1.jar:/home/jakob/.ivy2/cache/org.apache.zookeeper/zookeeper/jars/zookeeper-3.4.6.jar:/home/jakob/.ivy2/cache/org.bouncycastle/bcprov-jdk15on/jars/bcprov-jdk15on-1.51.jar:/home/jakob/.ivy2/cache/org.codehaus.jackson/jackson-core-asl/jars/jackson-core-asl-1.9.13.jar:/home/jakob/.ivy2/cache/org.codehaus.jackson/jackson-jaxrs/jars/jackson-jaxrs-1.9.13.jar:/home/jakob/.ivy2/cache/org.codehaus.jackson/jackson-mapper-asl/jars/jackson-mapper-asl-1.9.13.jar:/home/jakob/.ivy2/cache/org.codehaus.jackson/jackson-xc/jars/jackson-xc-1.9.13.jar:/home/jakob/.ivy2/cache/org.codehaus.janino/commons-compiler/jars/commons-compiler-3.0.0.jar:/home/jakob/.ivy2/cache/org.codehaus.janino/janino/jars/janino-3.0.0.jar:/home/jakob/.ivy2/cache/org.codehaus.jettison/jettison/bundles/jettison-1.1.jar:/home/jakob/.ivy2/cache/org.fusesource.leveldbjni/leveldbjni-all/bundles/leveldbjni-all-1.8.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2/hk2-api/jars/hk2-api-2.4.0-b34.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2/hk2-locator/jars/hk2-locator-2.4.0-b34.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2/hk2-utils/jars/hk2-utils-2.4.0-b34.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2/osgi-resource-locator/jars/osgi-resource-locator-1.0.1.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2.external/aopalliance-repackaged/jars/aopalliance-repackaged-2.4.0-b34.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2.external/javax.inject/jars/javax.inject-2.4.0-b34.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.bundles.repackaged/jersey-guava/bundles/jersey-guava-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.containers/jersey-container-servlet/jars/jersey-container-servlet-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.containers/jersey-container-servlet-core/jars/jersey-container-servlet-core-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.core/jersey-client/jars/jersey-client-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.core/jersey-common/jars/jersey-common-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.core/jersey-server/jars/jersey-server-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.media/jersey-media-jaxb/jars/jersey-media-jaxb-2.22.2.jar:/home/jakob/.ivy2/cache/org.htrace/htrace-core/jars/htrace-core-3.0.4.jar:/home/jakob/.ivy2/cache/org.javassist/javassist/bundles/javassist-3.18.1-GA.jar:/home/jakob/.ivy2/cache/org.jpmml/pmml-model/jars/pmml-model-1.2.15.jar:/home/jakob/.ivy2/cache/org.jpmml/pmml-schema/jars/pmml-schema-1.2.15.jar:/home/jakob/.ivy2/cache/org.json4s/json4s-ast_2.11/jars/json4s-ast_2.11-3.2.11.jar:/home/jakob/.ivy2/cache/org.json4s/json4s-core_2.11/jars/json4s-core_2.11-3.2.11.jar:/home/jakob/.ivy2/cache/org.json4s/json4s-jackson_2.11/jars/json4s-jackson_2.11-3.2.11.jar:/home/jakob/.ivy2/cache/org.mortbay.jetty/jetty-util/jars/jetty-util-6.1.26.jar:/home/jakob/.ivy2/cache/org.objenesis/objenesis/jars/objenesis-2.1.jar:/home/jakob/.ivy2/cache/org.roaringbitmap/RoaringBitmap/bundles/RoaringBitmap-0.5.11.jar:/home/jakob/.ivy2/cache/org.scala-lang/scala-compiler/jars/scala-compiler-2.11.8.jar:/home/jakob/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:/home/jakob/.ivy2/cache/org.scala-lang/scala-reflect/jars/scala-reflect-2.11.8.jar:/home/jakob/.ivy2/cache/org.scala-lang/scalap/jars/scalap-2.11.8.jar:/home/jakob/.ivy2/cache/org.scala-lang.modules/scala-parser-combinators_2.11/bundles/scala-parser-combinators_2.11-1.0.4.jar:/home/jakob/.ivy2/cache/org.scala-lang.modules/scala-xml_2.11/bundles/scala-xml_2.11-1.0.4.jar:/home/jakob/.ivy2/cache/org.scalanlp/breeze-macros_2.11/jars/breeze-macros_2.11-0.13.1.jar:/home/jakob/.ivy2/cache/org.scalanlp/breeze-natives_2.11/jars/breeze-natives_2.11-0.12.jar:/home/jakob/.ivy2/cache/org.scalanlp/breeze-viz_2.11/jars/breeze-viz_2.11-0.12.jar:/home/jakob/.ivy2/cache/org.scalanlp/breeze_2.11/jars/breeze_2.11-0.13.1.jar:/home/jakob/.ivy2/cache/org.slf4j/jcl-over-slf4j/jars/jcl-over-slf4j-1.7.16.jar:/home/jakob/.ivy2/cache/org.slf4j/jul-to-slf4j/jars/jul-to-slf4j-1.7.16.jar:/home/jakob/.ivy2/cache/org.slf4j/slf4j-api/jars/slf4j-api-1.7.16.jar:/home/jakob/.ivy2/cache/org.slf4j/slf4j-log4j12/jars/slf4j-log4j12-1.7.16.jar:/home/jakob/.ivy2/cache/org.sonatype.sisu.inject/cglib/jars/cglib-2.2.1-v20090111.jar:/home/jakob/.ivy2/cache/org.spark-project.spark/unused/jars/unused-1.0.0.jar:/home/jakob/.ivy2/cache/org.spire-math/spire-macros_2.11/jars/spire-macros_2.11-0.13.0.jar:/home/jakob/.ivy2/cache/org.spire-math/spire_2.11/jars/spire_2.11-0.13.0.jar:/home/jakob/.ivy2/cache/org.tukaani/xz/jars/xz-1.0.jar:/home/jakob/.ivy2/cache/org.typelevel/machinist_2.11/jars/machinist_2.11-0.6.1.jar:/home/jakob/.ivy2/cache/org.typelevel/macro-compat_2.11/jars/macro-compat_2.11-1.1.1.jar:/home/jakob/.ivy2/cache/org.xerial.snappy/snappy-java/bundles/snappy-java-1.1.2.6.jar:/home/jakob/.ivy2/cache/oro/oro/jars/oro-2.0.8.jar:/home/jakob/.ivy2/cache/xerces/xercesImpl/jars/xercesImpl-2.9.1.jar:/home/jakob/.ivy2/cache/xml-apis/xml-apis/jars/xml-apis-1.3.04.jar:/home/jakob/.ivy2/cache/xmlenc/xmlenc/jars/xmlenc-0.52.jar SVM.TestLocalAlgorithm
Empirical dataset from local file system with 0 variables.
Observations: 0 (training), 0 (test)
Data was not yet generated.

The input file /home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/magic04train.csv has 12680 lines and 12 columns.
Summary statistics of train data set before z-transformation:
mean:		variance:		standard deviation:
6339.5               1.339959E7            3660.545041383865
53.272122673501556   1818.0896606786725    42.63906261491536
22.11556955835961    339.1220152777037     18.41526582153252
2.8227191640378555   0.22337316786206698   0.47262370641141876
0.3809673580441642   0.03326486575435714   0.18238658326301618
0.21497101735015878  0.012109416777259746  0.11004279520831768
-4.8474820504732     3563.4296437166977    59.69446912165898
10.030569227129366   2636.1222805503703    51.34318144165173
0.27793725552050397  432.2682987920789     20.791062954839006
27.756945457413263   679.4403600041117     26.066076804999092
193.26211767350148   5561.9388184949885    74.57840718663137
Summary statistics of train data set AFTER z-transformation:
mean:	variance:	standard deviation:
-2.600160755109049E-14   1.0000000000001201  1.00000000000006
5.232190277825594E-16    0.999999999999996   0.999999999999998
6.369518356762888E-16    0.9999999999999987  0.9999999999999993
-1.2910510063367936E-15  0.9999999999999936  0.9999999999999968
-9.292883167621654E-16   1.0000000000000056  1.0000000000000027
-9.535524225692121E-15   0.999999999999997   0.9999999999999984
2.3889039618002483E-16   1.0000000000000016  1.0000000000000007
-5.493042544262028E-16   0.9999999999999964  0.9999999999999982
3.2912312198513094E-17   0.9999999999999986  0.9999999999999992
-5.391906810359864E-16   1.0000000000000036  1.0000000000000018
1.3268737237420725E-15   1.000000000000003   1.0000000000000016
The input file /home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/magic04test.csv has 6340 lines and 12 columns.
Summary statistics of test data set BEFORE z-transformation with means and standard deviation of the training set:
mean:	variance:	standard deviation:
3169.5               3350161.6666666665    1830.3446852073155
53.2062164353311     1748.4400415036146    41.81435209953174
22.311759542586692   331.5163448605896     18.20759030900546
2.829612555205062    0.22330565872544658   0.4725522814731155
0.37904649842271343  0.033735045494914885  0.1836710251915497
0.21402936908517417  0.012420428110344925  0.11144697443333725
-3.3002713722397643  3388.161702489047     58.20791786766683
11.575495993690856   2529.6048812468284    50.295177514815755
0.1933033596214506   436.8739352261841     20.901529494900224
27.423229116719217   685.3498372498752     26.179187100631587
194.9298440536277    5629.672452385667     75.03114321657152
Summary statistics of test data set AFTER z-transformation with means and standard deviation of the training set:
mean:	variance:	standard deviation:
-0.8659912565375991     0.2500197145336576  0.5000197141450101
-0.0015456774640085985  0.9616907676879619  0.9806583338186455
0.010653660182181082    0.9775724663263593  0.9887226437815406
0.01458536902334537     0.9996977741898604  0.9998488756756495
-0.010531803310779872   1.0141344247119384  1.0070424145545898
-0.008557109651773908   1.0256834279309952  1.0127603013206012
0.025918828008675277    0.9508148164124183  0.9750973368912553
0.03009020327883636     0.9595931493430994  0.9795882550046725
-0.004070686336859687   1.0106545782954117  1.005313174237467
-0.012802706874170629   1.008697565810967   1.0043393678488197
0.022362054152656026    1.0121780616617813  1.0060706047101173
Training: 8173(+)/4507(-)/12680(total)
Test: 4159(+)/2181(-)/6340(total)
The relative proportion [%] of matrix elements in submatrices is:
submatrix 1: 25 %
submatrix 2: 24 %
submatrix 3: 24 %
submatrix 4: 25 %
Preparing the hash map for the test set.
The relative proportion [%] of matrix elements in submatrices is:
submatrix 1: 25 %
submatrix 2: 24 %
submatrix 3: 25 %
submatrix 4: 24 %
Run gradient descent.
Jan 21, 2018 12:38:49 PM com.github.fommil.jni.JniLoader liberalLoad
INFO: successfully loaded /tmp/jniloader452844247147396705netlib-native_system-linux-x86_64.so
Momentum 0.150 Cross-validate sparsity.
Calculate Z matrix
Calculate predictions
Determine the optimal sparsity
Nr of correct predictions for test set: 5125/6340 with sparsity: 24
Run gradient descent.
Momentum 0.055 Cross-validate sparsity.
Calculate Z matrix
Calculate predictions
Determine the optimal sparsity
Nr of correct predictions for test set: 5132/6340 with sparsity: 14
Run gradient descent.
Momentum 0.000 Cross-validate sparsity.
Calculate Z matrix
Calculate predictions
Determine the optimal sparsity
Nr of correct predictions for test set: 5143/6340 with sparsity: 18

Process finished with exit code 0
*/