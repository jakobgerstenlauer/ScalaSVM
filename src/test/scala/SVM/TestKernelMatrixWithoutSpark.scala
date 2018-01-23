package test
import SVM._

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}

//Important flags for the Java virtual machine:
//Force the JVM to cache Integers up to dimensionality of K and S:
//-Djava.lang.Integer.IntegerCache.high=50000
//This way duplicate integers in the HashMaps are cached
// and memory footprint is significantly reduced! (Flyweight pattern)
//All flags:
//-server -XX:+UseConcMarkSweepGC -XX:+CMSIncrementalMode -XX:+CMSIncrementalPacing -XX:CMSIncrementalDutyCycleMin=0 -XX:CMSIncrementalDutyCycle=10 -XX:+UseCMSInitiatingOccupancyOnly - -XX:ThreadStackSize=300 -XX:MaxTenuringThreshold=0 -XX:SurvivorRatio=128 -XX:+UseTLAB -XX:+PrintGCDetails -Xms12288M  -Xmx12288M  -XX:NewSize=3072M  -XX:MaxNewSize=3072M -XX:ParallelGCThreads=4 -Djava.lang.Integer.IntegerCache.high=1000000 -verbose:gc -Xloggc:"/home/jakob/Documents/UPC/master_thesis/jvm/logs"
object TestKernelMatrixWithoutSpark extends App {
  /*** Measures the processing time of a given Scala command.
    * Source: http://biercoff.com/easily-measuring-code-execution-time-in-scala/
    * @param block The code block to execute.
    * @tparam R
    * @return
    */
  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

	val kernelPar = GaussianKernelParameter(1.0)
	println(kernelPar)
	val gaussianKernel = GaussianKernel(kernelPar)
	println(gaussianKernel)
	val N = 400000
  Utility.testJVMArgs(N/2)
	val dataProperties = DataParams(N = N, d = 10)
	println(dataProperties)
	val d = new SimData(dataProperties)
	println(d)
	d.simulate()
	println(d)

  //First find a value for epsilon that is manageable:
	//val probeMatrices = ProbeMatrices(d, gaussianKernel)
	//Number of non-sparse matrix elements with epsilon = 0.001:
	val epsilon = 0.01
	//val numElementsS =  probeMatrices.probeSparsity(Test, 0.001)
	//val numElementsK =  probeMatrices.probeSparsity(Train, 0.001)
  //println("Projected memory requirements for epsilon ="+epsilon+":")
  //Integer = 32 bits = 4 Byte
  //val intsPerKB = 256
  //println("Training matrix K: "+numElementsK/intsPerKB+"kB:")
  //println("Training matrix S: "+numElementsS/intsPerKB+"kB:")

  val lmf = LeanMatrixFactory(d, gaussianKernel, epsilon)
	val mp = ModelParams(C = 0.5, delta = 0.03)
	val alphas = new Alphas(N=N/2, mp)
	val ap = AlgoParams(maxIter = 30, batchProb = 0.8, learningRateDecline = 0.5, epsilon = epsilon, quantileAlphaClipping=0.00)
	var algo = NoMatrices(alphas, ap, mp, lmf, new ListBuffer[Future[(Int,Int,Int)]])
	var numInt = 0
	while(numInt < ap.maxIter && algo.getSparsity < 99.0){
		algo = algo.iterate(numInt)
		numInt += 1
	}
	val testSetAccuracy : Future[Int] = algo.predictOnTestSet()
	Await.result(testSetAccuracy, Duration(60,"minutes"))
 /* Synthetic dataset with 10 variables.
  Observations: 50000 (training), 50000(test)
  Data was already generated.

  Elapsed time: 563596072ns
  Sequential approach: The matrix has 50000 rows.
    The hash map has 49960 <key,value> pairs.
    Elapsed time: 425812077187ns
  The hash map has 49970 <key,value> pairs.
    Elapsed time: 1179510652053ns
    Train:42511/50000=85%,Test:38139/50000=76%,Sparsity:0%
Train:44358/50000=89%,Test:37603/50000=75%,Sparsity:7%
Train:44861/50000=90%,Test:36681/50000=73%,Sparsity:10%
Train:44990/50000=90%,Test:36286/50000=73%,Sparsity:11%
Train:45093/50000=90%,Test:35826/50000=72%,Sparsity:12%
Train:45037/50000=90%,Test:35372/50000=71%,Sparsity:13%
Train:44945/50000=90%,Test:35024/50000=70%,Sparsity:14%
Train:44734/50000=89%,Test:34535/50000=69%,Sparsity:15%
Train:44643/50000=89%,Test:34309/50000=69%,Sparsity:16%
Train:44521/50000=89%,Test:33982/50000=68%,Sparsity:17%
Train:44045/50000=88%,Test:33566/50000=67%,Sparsity:18%
Train:43632/50000=87%,Test:33340/50000=67%,Sparsity:18%
  */

	/*/usr/lib/jvm/java-8-oracle/bin/java -server -XX:+UseConcMarkSweepGC -XX:+CMSIncrementalMode -XX:+CMSIncrementalPacing -XX:CMSIncrementalDutyCycleMin=0 -XX:CMSIncrementalDutyCycle=10 -XX:+UseCMSInitiatingOccupancyOnly -XX:ThreadStackSize=300 -XX:MaxTenuringThreshold=0 -XX:SurvivorRatio=128 -XX:+UseTLAB -XX:+PrintGCDetails -Xms12288M -Xmx12288M -XX:NewSize=3072M -XX:MaxNewSize=3072M -XX:ParallelGCThreads=4 -Djava.lang.Integer.IntegerCache.high=1000000 -verbose:gc -Xloggc:/home/jakob/Documents/UPC/master_thesis/jvm/logs/LogJVM.log -javaagent:/home/jakob/idea-IC-173.3727.127/lib/idea_rt.jar=35816:/home/jakob/idea-IC-173.3727.127/bin -Dfile.encoding=UTF-8 -classpath /usr/lib/jvm/java-8-oracle/jre/lib/charsets.jar:/usr/lib/jvm/java-8-oracle/jre/lib/deploy.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/cldrdata.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/dnsns.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/jaccess.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/jfxrt.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/localedata.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/nashorn.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/sunec.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/sunjce_provider.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/sunpkcs11.jar:/usr/lib/jvm/java-8-oracle/jre/lib/ext/zipfs.jar:/usr/lib/jvm/java-8-oracle/jre/lib/javaws.jar:/usr/lib/jvm/java-8-oracle/jre/lib/jce.jar:/usr/lib/jvm/java-8-oracle/jre/lib/jfr.jar:/usr/lib/jvm/java-8-oracle/jre/lib/jfxswt.jar:/usr/lib/jvm/java-8-oracle/jre/lib/jsse.jar:/usr/lib/jvm/java-8-oracle/jre/lib/management-agent.jar:/usr/lib/jvm/java-8-oracle/jre/lib/plugin.jar:/usr/lib/jvm/java-8-oracle/jre/lib/resources.jar:/usr/lib/jvm/java-8-oracle/jre/lib/rt.jar:/home/jakob/IdeaProjects/Dist_Online_Svm/target/scala-2.11/test-classes:/home/jakob/IdeaProjects/Dist_Online_Svm/target/scala-2.11/classes:/home/jakob/.ivy2/cache/aopalliance/aopalliance/jars/aopalliance-1.0.jar:/home/jakob/.ivy2/cache/org.scalatest/scalatest_2.11/bundles/scalatest_2.11-3.0.4.jar:/home/jakob/.ivy2/cache/org.scalactic/scalactic_2.11/bundles/scalactic_2.11-3.0.4.jar:/home/jakob/.ivy2/cache/org.scala-lang.modules/scala-xml_2.11/bundles/scala-xml_2.11-1.0.5.jar:/home/jakob/.ivy2/cache/bouncycastle/bcmail-jdk14/jars/bcmail-jdk14-138.jar:/home/jakob/.ivy2/cache/bouncycastle/bcprov-jdk14/jars/bcprov-jdk14-138.jar:/home/jakob/.ivy2/cache/com.chuusai/shapeless_2.11/bundles/shapeless_2.11-2.3.2.jar:/home/jakob/.ivy2/cache/com.clearspring.analytics/stream/jars/stream-2.7.0.jar:/home/jakob/.ivy2/cache/com.esotericsoftware/kryo-shaded/bundles/kryo-shaded-3.0.3.jar:/home/jakob/.ivy2/cache/com.esotericsoftware/minlog/bundles/minlog-1.3.0.jar:/home/jakob/.ivy2/cache/com.fasterxml.jackson.core/jackson-annotations/bundles/jackson-annotations-2.6.5.jar:/home/jakob/.ivy2/cache/com.fasterxml.jackson.core/jackson-core/bundles/jackson-core-2.6.5.jar:/home/jakob/.ivy2/cache/com.fasterxml.jackson.core/jackson-databind/bundles/jackson-databind-2.6.5.jar:/home/jakob/.ivy2/cache/com.fasterxml.jackson.module/jackson-module-paranamer/bundles/jackson-module-paranamer-2.6.5.jar:/home/jakob/.ivy2/cache/com.fasterxml.jackson.module/jackson-module-scala_2.11/bundles/jackson-module-scala_2.11-2.6.5.jar:/home/jakob/.ivy2/cache/com.github.fommil/jniloader/jars/jniloader-1.1.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/core/jars/core-1.1.2.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/native_ref-java/jars/native_ref-java-1.1.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/native_system-java/jars/native_system-java-1.1.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-linux-armhf/jars/netlib-native_ref-linux-armhf-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-linux-i686/jars/netlib-native_ref-linux-i686-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-linux-x86_64/jars/netlib-native_ref-linux-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-osx-x86_64/jars/netlib-native_ref-osx-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-win-i686/jars/netlib-native_ref-win-i686-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_ref-win-x86_64/jars/netlib-native_ref-win-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-linux-armhf/jars/netlib-native_system-linux-armhf-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-linux-i686/jars/netlib-native_system-linux-i686-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-linux-x86_64/jars/netlib-native_system-linux-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-osx-x86_64/jars/netlib-native_system-osx-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-win-i686/jars/netlib-native_system-win-i686-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.fommil.netlib/netlib-native_system-win-x86_64/jars/netlib-native_system-win-x86_64-1.1-natives.jar:/home/jakob/.ivy2/cache/com.github.rwl/jtransforms/jars/jtransforms-2.4.0.jar:/home/jakob/.ivy2/cache/com.google.code.findbugs/jsr305/jars/jsr305-1.3.9.jar:/home/jakob/.ivy2/cache/com.google.code.gson/gson/jars/gson-2.2.4.jar:/home/jakob/.ivy2/cache/com.google.guava/guava/jars/guava-11.0.2.jar:/home/jakob/.ivy2/cache/com.google.inject/guice/jars/guice-3.0.jar:/home/jakob/.ivy2/cache/com.google.protobuf/protobuf-java/bundles/protobuf-java-2.5.0.jar:/home/jakob/.ivy2/cache/com.jamesmurty.utils/java-xmlbuilder/jars/java-xmlbuilder-1.0.jar:/home/jakob/.ivy2/cache/com.lowagie/itext/jars/itext-2.1.5.jar:/home/jakob/.ivy2/cache/com.ning/compress-lzf/bundles/compress-lzf-1.0.3.jar:/home/jakob/.ivy2/cache/com.thoughtworks.paranamer/paranamer/jars/paranamer-2.6.jar:/home/jakob/.ivy2/cache/com.twitter/chill-java/jars/chill-java-0.8.0.jar:/home/jakob/.ivy2/cache/com.twitter/chill_2.11/jars/chill_2.11-0.8.0.jar:/home/jakob/.ivy2/cache/com.univocity/univocity-parsers/jars/univocity-parsers-2.2.1.jar:/home/jakob/.ivy2/cache/commons-beanutils/commons-beanutils/jars/commons-beanutils-1.7.0.jar:/home/jakob/.ivy2/cache/commons-beanutils/commons-beanutils-core/jars/commons-beanutils-core-1.8.0.jar:/home/jakob/.ivy2/cache/commons-cli/commons-cli/jars/commons-cli-1.2.jar:/home/jakob/.ivy2/cache/commons-codec/commons-codec/jars/commons-codec-1.10.jar:/home/jakob/.ivy2/cache/commons-collections/commons-collections/jars/commons-collections-3.2.2.jar:/home/jakob/.ivy2/cache/commons-configuration/commons-configuration/jars/commons-configuration-1.6.jar:/home/jakob/.ivy2/cache/commons-digester/commons-digester/jars/commons-digester-1.8.jar:/home/jakob/.ivy2/cache/commons-httpclient/commons-httpclient/jars/commons-httpclient-3.1.jar:/home/jakob/.ivy2/cache/commons-io/commons-io/jars/commons-io-2.4.jar:/home/jakob/.ivy2/cache/commons-lang/commons-lang/jars/commons-lang-2.6.jar:/home/jakob/.ivy2/cache/commons-logging/commons-logging/jars/commons-logging-1.0.4.jar:/home/jakob/.ivy2/cache/commons-net/commons-net/jars/commons-net-2.2.jar:/home/jakob/.ivy2/cache/io.dropwizard.metrics/metrics-core/bundles/metrics-core-3.1.2.jar:/home/jakob/.ivy2/cache/io.dropwizard.metrics/metrics-graphite/bundles/metrics-graphite-3.1.2.jar:/home/jakob/.ivy2/cache/io.dropwizard.metrics/metrics-json/bundles/metrics-json-3.1.2.jar:/home/jakob/.ivy2/cache/io.dropwizard.metrics/metrics-jvm/bundles/metrics-jvm-3.1.2.jar:/home/jakob/.ivy2/cache/io.netty/netty/bundles/netty-3.9.9.Final.jar:/home/jakob/.ivy2/cache/io.netty/netty-all/jars/netty-all-4.0.43.Final.jar:/home/jakob/.ivy2/cache/javax.activation/activation/jars/activation-1.1.1.jar:/home/jakob/.ivy2/cache/javax.annotation/javax.annotation-api/jars/javax.annotation-api-1.2.jar:/home/jakob/.ivy2/cache/javax.inject/javax.inject/jars/javax.inject-1.jar:/home/jakob/.ivy2/cache/javax.mail/mail/jars/mail-1.4.7.jar:/home/jakob/.ivy2/cache/javax.servlet/javax.servlet-api/jars/javax.servlet-api-3.1.0.jar:/home/jakob/.ivy2/cache/javax.validation/validation-api/jars/validation-api-1.1.0.Final.jar:/home/jakob/.ivy2/cache/javax.ws.rs/javax.ws.rs-api/jars/javax.ws.rs-api-2.0.1.jar:/home/jakob/.ivy2/cache/javax.xml.bind/jaxb-api/jars/jaxb-api-2.2.2.jar:/home/jakob/.ivy2/cache/javax.xml.stream/stax-api/jars/stax-api-1.0-2.jar:/home/jakob/.ivy2/cache/jfree/jcommon/jars/jcommon-1.0.16.jar:/home/jakob/.ivy2/cache/jfree/jfreechart/jars/jfreechart-1.0.13.jar:/home/jakob/.ivy2/cache/jline/jline/jars/jline-0.9.94.jar:/home/jakob/.ivy2/cache/log4j/log4j/bundles/log4j-1.2.17.jar:/home/jakob/.ivy2/cache/mx4j/mx4j/jars/mx4j-3.0.2.jar:/home/jakob/.ivy2/cache/net.iharder/base64/jars/base64-2.3.8.jar:/home/jakob/.ivy2/cache/net.java.dev.jets3t/jets3t/jars/jets3t-0.9.3.jar:/home/jakob/.ivy2/cache/net.jpountz.lz4/lz4/jars/lz4-1.3.0.jar:/home/jakob/.ivy2/cache/net.razorvine/pyrolite/jars/pyrolite-4.13.jar:/home/jakob/.ivy2/cache/net.sf.opencsv/opencsv/jars/opencsv-2.3.jar:/home/jakob/.ivy2/cache/net.sf.py4j/py4j/jars/py4j-0.10.4.jar:/home/jakob/.ivy2/cache/net.sourceforge.f2j/arpack_combined_all/jars/arpack_combined_all-0.1-javadoc.jar:/home/jakob/.ivy2/cache/net.sourceforge.f2j/arpack_combined_all/jars/arpack_combined_all-0.1.jar:/home/jakob/.ivy2/cache/org.antlr/antlr4-runtime/jars/antlr4-runtime-4.5.3.jar:/home/jakob/.ivy2/cache/org.apache.avro/avro/jars/avro-1.7.7.jar:/home/jakob/.ivy2/cache/org.apache.avro/avro-ipc/jars/avro-ipc-1.7.7.jar:/home/jakob/.ivy2/cache/org.apache.avro/avro-ipc/jars/avro-ipc-1.7.7-tests.jar:/home/jakob/.ivy2/cache/org.apache.avro/avro-mapred/jars/avro-mapred-1.7.7-hadoop2.jar:/home/jakob/.ivy2/cache/org.apache.commons/commons-compress/jars/commons-compress-1.4.1.jar:/home/jakob/.ivy2/cache/org.apache.commons/commons-crypto/jars/commons-crypto-1.0.0.jar:/home/jakob/.ivy2/cache/org.apache.commons/commons-lang3/jars/commons-lang3-3.5.jar:/home/jakob/.ivy2/cache/org.apache.commons/commons-math3/jars/commons-math3-3.4.1.jar:/home/jakob/.ivy2/cache/org.apache.curator/curator-client/bundles/curator-client-2.6.0.jar:/home/jakob/.ivy2/cache/org.apache.curator/curator-framework/bundles/curator-framework-2.6.0.jar:/home/jakob/.ivy2/cache/org.apache.curator/curator-recipes/bundles/curator-recipes-2.6.0.jar:/home/jakob/.ivy2/cache/org.apache.directory.api/api-asn1-api/bundles/api-asn1-api-1.0.0-M20.jar:/home/jakob/.ivy2/cache/org.apache.directory.api/api-util/bundles/api-util-1.0.0-M20.jar:/home/jakob/.ivy2/cache/org.apache.directory.server/apacheds-i18n/bundles/apacheds-i18n-2.0.0-M15.jar:/home/jakob/.ivy2/cache/org.apache.directory.server/apacheds-kerberos-codec/bundles/apacheds-kerberos-codec-2.0.0-M15.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-annotations/jars/hadoop-annotations-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-auth/jars/hadoop-auth-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-client/jars/hadoop-client-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-common/jars/hadoop-common-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-hdfs/jars/hadoop-hdfs-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-mapreduce-client-app/jars/hadoop-mapreduce-client-app-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-mapreduce-client-common/jars/hadoop-mapreduce-client-common-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-mapreduce-client-core/jars/hadoop-mapreduce-client-core-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-mapreduce-client-jobclient/jars/hadoop-mapreduce-client-jobclient-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-mapreduce-client-shuffle/jars/hadoop-mapreduce-client-shuffle-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-yarn-api/jars/hadoop-yarn-api-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-yarn-client/jars/hadoop-yarn-client-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-yarn-common/jars/hadoop-yarn-common-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.hadoop/hadoop-yarn-server-common/jars/hadoop-yarn-server-common-2.6.5.jar:/home/jakob/.ivy2/cache/org.apache.httpcomponents/httpclient/jars/httpclient-4.3.6.jar:/home/jakob/.ivy2/cache/org.apache.httpcomponents/httpcore/jars/httpcore-4.3.3.jar:/home/jakob/.ivy2/cache/org.apache.ivy/ivy/jars/ivy-2.4.0.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-column/jars/parquet-column-1.8.2.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-common/jars/parquet-common-1.8.2.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-encoding/jars/parquet-encoding-1.8.2.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-format/jars/parquet-format-2.3.1.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-hadoop/jars/parquet-hadoop-1.8.2.jar:/home/jakob/.ivy2/cache/org.apache.parquet/parquet-jackson/jars/parquet-jackson-1.8.2.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-catalyst_2.11/jars/spark-catalyst_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-core_2.11/jars/spark-core_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-graphx_2.11/jars/spark-graphx_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-launcher_2.11/jars/spark-launcher_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-mllib-local_2.11/jars/spark-mllib-local_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-mllib_2.11/jars/spark-mllib_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-network-common_2.11/jars/spark-network-common_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-network-shuffle_2.11/jars/spark-network-shuffle_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-sketch_2.11/jars/spark-sketch_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-sql_2.11/jars/spark-sql_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-streaming_2.11/jars/spark-streaming_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-tags_2.11/jars/spark-tags_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.spark/spark-unsafe_2.11/jars/spark-unsafe_2.11-2.2.0.jar:/home/jakob/.ivy2/cache/org.apache.xbean/xbean-asm5-shaded/bundles/xbean-asm5-shaded-4.4.jar:/home/jakob/.ivy2/cache/org.apache.xmlgraphics/xmlgraphics-commons/jars/xmlgraphics-commons-1.3.1.jar:/home/jakob/.ivy2/cache/org.apache.zookeeper/zookeeper/jars/zookeeper-3.4.6.jar:/home/jakob/.ivy2/cache/org.bouncycastle/bcprov-jdk15on/jars/bcprov-jdk15on-1.51.jar:/home/jakob/.ivy2/cache/org.codehaus.jackson/jackson-core-asl/jars/jackson-core-asl-1.9.13.jar:/home/jakob/.ivy2/cache/org.codehaus.jackson/jackson-jaxrs/jars/jackson-jaxrs-1.9.13.jar:/home/jakob/.ivy2/cache/org.codehaus.jackson/jackson-mapper-asl/jars/jackson-mapper-asl-1.9.13.jar:/home/jakob/.ivy2/cache/org.codehaus.jackson/jackson-xc/jars/jackson-xc-1.9.13.jar:/home/jakob/.ivy2/cache/org.codehaus.janino/commons-compiler/jars/commons-compiler-3.0.0.jar:/home/jakob/.ivy2/cache/org.codehaus.janino/janino/jars/janino-3.0.0.jar:/home/jakob/.ivy2/cache/org.codehaus.jettison/jettison/bundles/jettison-1.1.jar:/home/jakob/.ivy2/cache/org.fusesource.leveldbjni/leveldbjni-all/bundles/leveldbjni-all-1.8.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2/hk2-api/jars/hk2-api-2.4.0-b34.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2/hk2-locator/jars/hk2-locator-2.4.0-b34.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2/hk2-utils/jars/hk2-utils-2.4.0-b34.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2/osgi-resource-locator/jars/osgi-resource-locator-1.0.1.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2.external/aopalliance-repackaged/jars/aopalliance-repackaged-2.4.0-b34.jar:/home/jakob/.ivy2/cache/org.glassfish.hk2.external/javax.inject/jars/javax.inject-2.4.0-b34.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.bundles.repackaged/jersey-guava/bundles/jersey-guava-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.containers/jersey-container-servlet/jars/jersey-container-servlet-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.containers/jersey-container-servlet-core/jars/jersey-container-servlet-core-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.core/jersey-client/jars/jersey-client-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.core/jersey-common/jars/jersey-common-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.core/jersey-server/jars/jersey-server-2.22.2.jar:/home/jakob/.ivy2/cache/org.glassfish.jersey.media/jersey-media-jaxb/jars/jersey-media-jaxb-2.22.2.jar:/home/jakob/.ivy2/cache/org.htrace/htrace-core/jars/htrace-core-3.0.4.jar:/home/jakob/.ivy2/cache/org.javassist/javassist/bundles/javassist-3.18.1-GA.jar:/home/jakob/.ivy2/cache/org.jpmml/pmml-model/jars/pmml-model-1.2.15.jar:/home/jakob/.ivy2/cache/org.jpmml/pmml-schema/jars/pmml-schema-1.2.15.jar:/home/jakob/.ivy2/cache/org.json4s/json4s-ast_2.11/jars/json4s-ast_2.11-3.2.11.jar:/home/jakob/.ivy2/cache/org.json4s/json4s-core_2.11/jars/json4s-core_2.11-3.2.11.jar:/home/jakob/.ivy2/cache/org.json4s/json4s-jackson_2.11/jars/json4s-jackson_2.11-3.2.11.jar:/home/jakob/.ivy2/cache/org.mortbay.jetty/jetty-util/jars/jetty-util-6.1.26.jar:/home/jakob/.ivy2/cache/org.objenesis/objenesis/jars/objenesis-2.1.jar:/home/jakob/.ivy2/cache/org.roaringbitmap/RoaringBitmap/bundles/RoaringBitmap-0.5.11.jar:/home/jakob/.ivy2/cache/org.scala-lang/scala-compiler/jars/scala-compiler-2.11.8.jar:/home/jakob/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:/home/jakob/.ivy2/cache/org.scala-lang/scala-reflect/jars/scala-reflect-2.11.8.jar:/home/jakob/.ivy2/cache/org.scala-lang/scalap/jars/scalap-2.11.8.jar:/home/jakob/.ivy2/cache/org.scala-lang.modules/scala-parser-combinators_2.11/bundles/scala-parser-combinators_2.11-1.0.4.jar:/home/jakob/.ivy2/cache/org.scala-lang.modules/scala-xml_2.11/bundles/scala-xml_2.11-1.0.4.jar:/home/jakob/.ivy2/cache/org.scalanlp/breeze-macros_2.11/jars/breeze-macros_2.11-0.13.1.jar:/home/jakob/.ivy2/cache/org.scalanlp/breeze-natives_2.11/jars/breeze-natives_2.11-0.12.jar:/home/jakob/.ivy2/cache/org.scalanlp/breeze-viz_2.11/jars/breeze-viz_2.11-0.12.jar:/home/jakob/.ivy2/cache/org.scalanlp/breeze_2.11/jars/breeze_2.11-0.13.1.jar:/home/jakob/.ivy2/cache/org.slf4j/jcl-over-slf4j/jars/jcl-over-slf4j-1.7.16.jar:/home/jakob/.ivy2/cache/org.slf4j/jul-to-slf4j/jars/jul-to-slf4j-1.7.16.jar:/home/jakob/.ivy2/cache/org.slf4j/slf4j-api/jars/slf4j-api-1.7.16.jar:/home/jakob/.ivy2/cache/org.slf4j/slf4j-log4j12/jars/slf4j-log4j12-1.7.16.jar:/home/jakob/.ivy2/cache/org.sonatype.sisu.inject/cglib/jars/cglib-2.2.1-v20090111.jar:/home/jakob/.ivy2/cache/org.spark-project.spark/unused/jars/unused-1.0.0.jar:/home/jakob/.ivy2/cache/org.spire-math/spire-macros_2.11/jars/spire-macros_2.11-0.13.0.jar:/home/jakob/.ivy2/cache/org.spire-math/spire_2.11/jars/spire_2.11-0.13.0.jar:/home/jakob/.ivy2/cache/org.tukaani/xz/jars/xz-1.0.jar:/home/jakob/.ivy2/cache/org.typelevel/machinist_2.11/jars/machinist_2.11-0.6.1.jar:/home/jakob/.ivy2/cache/org.typelevel/macro-compat_2.11/jars/macro-compat_2.11-1.1.1.jar:/home/jakob/.ivy2/cache/org.xerial.snappy/snappy-java/bundles/snappy-java-1.1.2.6.jar:/home/jakob/.ivy2/cache/oro/oro/jars/oro-2.0.8.jar:/home/jakob/.ivy2/cache/xerces/xercesImpl/jars/xercesImpl-2.9.1.jar:/home/jakob/.ivy2/cache/xml-apis/xml-apis/jars/xml-apis-1.3.04.jar:/home/jakob/.ivy2/cache/xmlenc/xmlenc/jars/xmlenc-0.52.jar test.TestKernelMatrixWithoutSpark
	Java HotSpot(TM) 64-Bit Server VM warning: Using incremental CMS is deprecated and will likely be removed in a future release
		Gaussian kernel parameter sigma 1.0.

	Gaussian kernel:
		sigma: 1.0

	The argument IntegerCache.high is defined as -Djava.lang.Integer.IntegerCache.high=1000000
	Data parameters:
		Total number of observations: 200000
	Observations training set: 100000
	Observations validation set: 60000
	Observations test set: 40000
	Number of features: 10

	Synthetic dataset with 10 variables.
	Observations: 100000 (training), 60000(validation), 40000(test)
	Data was not yet generated.

		Jan 23, 2018 6:35:35 PM com.github.fommil.jni.JniLoader liberalLoad
		INFO: successfully loaded /tmp/jniloader116736691305923800netlib-native_system-linux-x86_64.so
	Jan 23, 2018 6:35:35 PM com.github.fommil.jni.JniLoader load
		INFO: already loaded netlib-native_system-linux-x86_64.so
	Synthetic dataset with 10 variables.
	Observations: 100000 (training), 60000(validation), 40000(test)
	Data was already generated.

	Momentum 0.144 Accuracy validation set: 49681/60000 with sparsity: 6
	Momentum 0.023 Accuracy validation set: 49609/60000 with sparsity: 1
	Momentum 0.000 Accuracy validation set: 49503/60000 with sparsity: 7
	Based on cross-validation, the optimal sparsity of: 6 with max correct predictions: 49681 was achieved in iteration: 0
	Predict on the test set.
		Nr of correct predictions for test set: 32879/40000 = 0.821975% accuracy


	Process finished with exit code 0

	This took 60 minutes and needed max 9 GB.
	Note that epsilon was 0.0001!
	*/

}